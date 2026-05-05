"""Model Context Protocol (MCP) definitions and client interface."""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Callable, Awaitable, Optional, Tuple
from pydantic import BaseModel, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPToolSchema(BaseModel):
    """JSON Schema for the tool's input arguments."""
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, model: Any) -> "MCPToolSchema":
        """Generates an MCPToolSchema from a Pydantic BaseModel."""
        schema = model.model_json_schema()
        return cls(
            type=schema.get("type", "object"),
            properties=schema.get("properties", {}),
            required=schema.get("required", [])
        )

class MCPTool(BaseModel):
    """Metadata for an MCP Tool."""
    name: str
    description: str
    inputSchema: MCPToolSchema
    source: str = "internal" # "internal" or name of external server

class MCPClient:
    """A client to manage and execute MCP tools from internal and external sources."""
    
    def __init__(self):
        # Internal tools: {name: (MCPTool, handler)}
        self._internal_tools: Dict[str, tuple[MCPTool, Callable[..., Awaitable[Any]]]] = {}
        # External sessions: {server_name: ClientSession}
        self._external_sessions: Dict[str, ClientSession] = {}
        # External tools mapping: {tool_name: server_name}
        self._external_tool_map: Dict[str, str] = {}
        # To keep track of external tool metadata
        self._external_tool_metadata: Dict[str, MCPTool] = {}
        
        self._lock = asyncio.Lock()
        self._initialized = False
        self._shutdown = False

    def register_tool(self, tool: MCPTool, handler: Callable[..., Awaitable[Any]]):
        """Registers an internal tool."""
        self._internal_tools[tool.name] = (tool, handler)
        logger.info(f"Registered internal MCP tool: {tool.name}")

    async def initialize_external_servers(self, server_configs: Dict[str, Any]):
        """Starts external MCP servers and initializes sessions."""
        async with self._lock:
            if self._initialized:
                return
            
            for name, config in server_configs.items():
                try:
                    command = config.get("command")
                    args = config.get("args", [])
                    env = config.get("env", os.environ.copy())
                    
                    if not command:
                        logger.warning(f"No command specified for MCP server {name}")
                        continue
                        
                    params = StdioServerParameters(
                        command=command,
                        args=args,
                        env=env
                    )
                    
                    # We start a background task for each server's connection
                    asyncio.create_task(self._connect_to_server(name, params))
                    
                except Exception as e:
                    logger.error(f"Failed to setup external MCP server {name}: {e}")
            
            self._initialized = True

    async def _connect_to_server(self, name: str, params: StdioServerParameters):
        """Internal helper to connect to a single MCP server with reconnection logic."""
        retry_delay = 1
        max_delay = 60
        
        while not self._shutdown:
            try:
                logger.info(f"Connecting to external MCP server {name}...")
                
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        self._external_sessions[name] = session
                        
                        # Discover tools from this server
                        tools_response = await session.list_tools()
                        for tool in tools_response.tools:
                            full_name = tool.name # We might want to namespace this if collisions occur
                            self._external_tool_map[full_name] = name
                            self._external_tool_metadata[full_name] = MCPTool(
                                name=tool.name,
                                description=tool.description,
                                inputSchema=MCPToolSchema(**tool.inputSchema),
                                source=name
                            )
                            logger.info(f"Registered external tool '{full_name}' from server '{name}'")
                        
                        retry_delay = 1 # Reset delay on successful connection
                        
                        # Keep the session alive until the application shuts down or connection fails
                        while not self._shutdown and name in self._external_sessions:
                            await asyncio.sleep(1)
                            
            except Exception as e:
                if not self._shutdown:
                    logger.error(f"Error in MCP server session '{name}': {e}")
                
            # Cleanup on disconnect
            self._external_sessions.pop(name, None)
            tools_to_remove = [tname for tname, sname in self._external_tool_map.items() if sname == name]
            for tname in tools_to_remove:
                self._external_tool_map.pop(tname, None)
                self._external_tool_metadata.pop(tname, None)
                
            if self._shutdown:
                break
                
            logger.info(f"Retrying connection to '{name}' in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_delay)

    def list_tools(self) -> List[MCPTool]:
        """Returns a list of all registered tools (internal + external)."""
        tools = [t[0] for t in self._internal_tools.values()]
        tools.extend(self._external_tool_metadata.values())
        return tools

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Returns tools in the format expected by the LLM."""
        llm_tools = []
        
        # Internal tools
        for tool, _ in self._internal_tools.values():
            llm_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema.model_dump()
                }
            })
            
        # External tools
        for tool in self._external_tool_metadata.values():
            llm_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema.model_dump()
                }
            })
            
        return llm_tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Calls a tool by name, routing to internal or external sources."""
        # 1. Check internal tools
        if name in self._internal_tools:
            _, handler = self._internal_tools[name]
            return await handler(**arguments)
            
        # 2. Check external tools
        if name in self._external_tool_map:
            server_name = self._external_tool_map[name]
            session = self._external_sessions.get(server_name)
            if not session:
                raise ValueError(f"MCP server '{server_name}' is not connected.")
            
            logger.info(f"Calling external MCP tool '{name}' on server '{server_name}'")
            return await session.call_tool(name, arguments)
            
        raise ValueError(f"Unknown MCP tool: {name}")

    async def shutdown(self):
        """Closes all external sessions."""
        async with self._lock:
            self._shutdown = True
            names = list(self._external_sessions.keys())
            for name in names:
                session = self._external_sessions.pop(name)
                # ClientSession doesn't have an explicit close in some versions, 
                # but the context manager handles it.
                # Here we just remove it from our tracking.
                logger.info(f"Shutting down MCP server session '{name}'")

# Global MCP client instance
mcp_client = MCPClient()
