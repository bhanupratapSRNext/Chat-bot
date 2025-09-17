# PDF Chat Bot

A Flask-based chatbot application that uses LangChain, Pinecone, and OpenAI to create an intelligent conversational interface for PDF documents. The bot can understand and answer questions about the content of uploaded PDF files using semantic search and natural language processing.

## Features

- PDF document processing and text extraction
- Semantic text chunking for better context understanding
- Vector embeddings using Hugging Face's sentence transformers
- Vector storage and similarity search using Pinecone
- Conversational AI powered by OpenAI's language models
- Flask web interface for easy interaction
- Efficient document indexing with duplicate prevention

## Technology Stack

- **Backend Framework**: Flask
- **Language Models**: 
  - OpenAI GPT for conversation
  - Hugging Face sentence-transformers for embeddings
- **Vector Database**: Pinecone
- **Document Processing**: LangChain
- **PDF Processing**: PyPDF

## Prerequisites

Before running the application, make sure you have:

- Python 3.8 or higher
- A Pinecone API key
- An OpenAI API key
- The PDF documents you want to process

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bhanupratapSRNext/Chat-bot.git
cd Chat-bot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
PINECONE_INDEX_NAME=your_index_name
```

## Project Structure

```
Chat-bot/
├── app.py                 # Main Flask application
├── requirements.txt       # Project dependencies
├── Blueprints/
│   ├── helper.py         # Utility functions for PDF processing and embeddings
│   ├── prompt.py         # Chatbot prompt templates
│   └── pinecone_index.py # Pinecone index management
├── Data/                 # Directory for PDF documents
└── test/                # Test files and notebooks
```

## How It Works

1. **Document Processing**:
   - PDF files in the `Data` directory are loaded and processed
   - Documents are split into semantic chunks for better context understanding

2. **Vector Storage**:
   - Text chunks are converted to embeddings using Hugging Face's sentence transformers
   - Embeddings are stored in Pinecone for efficient similarity search
   - The system checks for existing indexes to prevent duplicate processing

3. **Query Processing**:
   - User queries are processed through the conversational chain
   - Relevant context is retrieved from Pinecone using similarity search
   - OpenAI's language model generates contextual responses

## Environment Variables

- `PINECONE_API_KEY`: Your Pinecone API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_INDEX_NAME`: Name of your Pinecone index

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Author

bhanupratapSRNext

## Acknowledgments

- LangChain for the excellent document processing tools
- Pinecone for vector similarity search capabilities
- OpenAI for the language model
- Hugging Face for the sentence transformers
