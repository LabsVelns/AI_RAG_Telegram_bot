from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import os
from rag_pipeline import ask  # your RAG function
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("AI_RAG_SIMPLE_bot")  # Set your bot token as an environment variable
user_memory = {} 

if not os.path.exists("vectorstore"):
    print("Building vector DB...")
    os.system("python ingestion.py")
    
# 🔹 /start
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 AI RAG Bot Ready!\n\nUse /ask <your question>"
    )


# 🔹 /ask
def ask_question(update: Update, context: CallbackContext):
    query = " ".join(context.args)
    chat_id = update.message.chat_id

    if chat_id not in user_memory:
        user_memory[chat_id] = []

    if not query:
        update.message.reply_text("❗ Please provide a question.\nExample: /ask What is deep learning?")
        return

    update.message.reply_text("⏳ Thinking...")
    # store user query
    user_memory[chat_id].append(query)

    # keep last 3
    user_memory[chat_id] = user_memory[chat_id][-3:]

    try:
        answer, docs = ask(query, history=user_memory[chat_id])

        sources = "\n".join([
            f"{doc.metadata['source']} (p{doc.metadata['page']})"
            for doc in docs
        ])

        response = f"""
📌 Answer:
{answer}

📚 Sources:
{sources}
"""

        update.message.reply_text(response)

    except Exception as e:
        update.message.reply_text("❌ Error occurred.")


# 🔹 main
def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("ask", ask_question))

    print("🤖 Bot running...")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()