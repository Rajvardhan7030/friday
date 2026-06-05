import asyncio
import ollama

async def main():
    client = ollama.AsyncClient()
    response = await client.chat(
        model='llama3.2', 
        messages=[{'role':'user', 'content':'What is the weather in New York?'}], 
        tools=[{
            'type': 'function', 
            'function': {
                'name': 'get_weather', 
                'description': 'Get weather', 
                'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}, 'required': ['location']}
            }
        }], 
        stream=True
    )
    async for chunk in response:
        print(chunk)

asyncio.run(main())
