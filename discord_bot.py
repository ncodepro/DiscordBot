# bot.py
import os
import discord
from reader import WebpageQATool
from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

# Get the Discord bot token from an environment variable for security reasons
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# We set the intents that our bot needs. Here, we need to access message content.
intents = discord.Intents.default()
intents.message_content = True

# Create a client instance for our bot
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    """Event that triggers when the bot has successfully connected to Discord"""
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    """Event that triggers when a message is received by the bot"""
    # Check if the bot is mentioned in the message
    if client.user.mentioned_in(message):
        # Initialize the ChatOpenAI model
        llm = ChatOpenAI(temperature=1.0)

        # Initialize the WebpageQATool with the QA chain
        query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))

        # Extract question and url from the message
        data = message.clean_content.split(',')
        if len(data) != 2:
            await message.reply("Please provide a question and a URL.")
            return

        question, url = data

        # Run the WebpageQATool
        output = query_website_tool.run(question+","+url)["output_text"]

        # Reply to the message with the output of the WebpageQATool
        await message.reply(output)

# Run the bot using the token
client.run(TOKEN)
