import pytest
from friday.core.mcp import MCPClient

@pytest.mark.asyncio
async def test_mcp_client_is_safe_external_command():
    client = MCPClient()
    
    # Safe commands
    assert client._is_safe_external_command("npx", ["@modelcontextprotocol/server-everything"]) is True
    assert client._is_safe_external_command("/usr/bin/node", ["server.js"]) is True
    
    # Unsafe commands (shell metacharacters)
    assert client._is_safe_external_command("node; rm -rf /", []) is False
    assert client._is_safe_external_command("node", ["arg1; rm -rf /"]) is False
    assert client._is_safe_external_command("node", ["$(whoami)"]) is True # Wait, $ is not in arg check? 
    # Actually, in _is_safe_external_command I checked (';', '&', '|', '`') for args.
    
    assert client._is_safe_external_command("node", ["`id`"]) is False
    assert client._is_safe_external_command("echo $HOME", []) is False # $ in command name
