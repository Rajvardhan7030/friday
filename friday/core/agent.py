import logging
from typing import Optional
from pydantic import BaseModel
from friday.core.memory import ConversationMemory, SemanticMemory
from friday.core.skill_manager import SkillManager
from friday.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class UserMessage(BaseModel):
    content: str
    platform: str
    user_id: str
    timestamp: float

class FridayAgent:
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        conv_memory: ConversationMemory,
        semantic_memory: SemanticMemory,
        skill_manager: SkillManager
    ):
        self.llm = llm_provider
        self.conv_memory = conv_memory
        self.semantic_memory = semantic_memory
        self.skill_manager = skill_manager

    async def process(self, message: UserMessage) -> str:
        # 1. RECALL relevant memory
        try:
            query_embed = await self.llm.embed(message.content)
            recalls = await self.semantic_memory.recall(query_embed, k=3)
            memory_context = "\n".join([r["document"] for r in recalls])
        except Exception as e:
            logger.warning(f"Failed to recall memories: {e}")
            memory_context = ""

        # Fetch recent chat context
        conversation_id = f"{message.platform}_{message.user_id}"
        await self.conv_memory.add_message(conversation_id, "user", message.content, message.platform)
        recent_msgs = await self.conv_memory.get_messages(conversation_id, limit=10)

        # 2. REASON: build the prompt and decide on action
        skill_list = self.skill_manager.get_skill_list()
        system_prompt = (
            "You are FRIDAY, a self-improving personal assistant.\n"
            f"You have access to the following skills:\n{skill_list}\n\n"
            "If the user asks you to perform a task that matches a skill, output ONLY the string: 'EXECUTE: <skill_name>'.\n"
            "If no skill exists for the task, generate one using the skill_creation protocol, or just reply naturally to the chat.\n"
            "To generate a skill, reply with 'CREATE_SKILL: <name> | <description> | <code>'.\n"
            f"Past Context:\n{memory_context}\n"
        )

        messages_for_llm = [{"role": "system", "content": system_prompt}]
        for msg in recent_msgs:
            messages_for_llm.append({"role": msg["role"], "content": msg["content"]})

        response = await self.llm.chat(messages_for_llm)

        # 3. ACT: execute skill or just respond
        final_response = response
        if response.startswith("EXECUTE:"):
            parts = response.split("EXECUTE:")
            skill_name = parts[1].strip()
            final_response = await self.skill_manager.execute_skill(skill_name)
        elif response.startswith("CREATE_SKILL:"):
            try:
                parts = response.split("|", 2)
                name = parts[0].replace("CREATE_SKILL:", "").strip()
                desc = parts[1].strip()
                code = parts[2].strip()
                self.skill_manager.create_skill(name, code, desc, {})
                final_response = f"Skill '{name}' created successfully."
            except Exception as e:
                final_response = f"Failed to create skill: {e}"

        # 4. LEARN: Store to memory
        await self.conv_memory.add_message(conversation_id, "assistant", final_response, message.platform)
        
        try:
            combined_text = f"User: {message.content}\nFRIDAY: {final_response}"
            summary_embed = await self.llm.embed(combined_text)
            await self.semantic_memory.remember(
                text=combined_text,
                embedding=summary_embed,
                metadata={"platform": message.platform, "user_id": message.user_id, "timestamp": message.timestamp}
            )
        except Exception as e:
            logger.warning(f"Failed to learn from conversation: {e}")

        return final_response
