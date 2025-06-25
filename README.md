# WebLLM - AI-Powered Web Interface

![Project Logo](https://via.placeholder.com/150) *(Replace with actual logo URL)*

WebLLM is a powerful web-based interface for interacting with various AI models, including Gemini, OpenAI, Claude, Mistral, and Grok. Built with Python and Flask, it provides a seamless experience for generating text and analyzing images.

## Features

- **Multi-Model Support**: Switch between different AI models effortlessly
- **Text Generation**: Generate high-quality text content
- **Image Analysis**: Upload and analyze images using AI
- **Streaming Responses**: Get real-time streaming responses for long prompts
- **Customizable**: Easily configure API keys and model preferences

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Xenonesis/webllm.git
   cd webllm
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys:
   - Edit `config.json` with your API credentials
   - Set up at least one AI provider (Gemini recommended)

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the web interface at `http://localhost:5000`

## Configuration

Edit `config.json` to set up your preferences:

```json
{
  "openai_api_key": "your_openai_api_key_here",
  "gemini_api_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=YOUR_API_KEY",
  "default_model": "gemini"
}
```

## Usage

1. Select your preferred AI model from the dropdown
2. Enter your prompt in the text area
3. Choose between text or image mode
4. Click "Generate" and view the results

For image analysis:
1. Upload an image file
2. Add an optional prompt for specific analysis
3. View the AI's interpretation of the image

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version History

- **v0.10** (2025-06-26): Updated Gemini API to 1.5-flash
- **v0.9** (2025-06-25): Initial release with basic functionality

## Support

For support, please open an issue on [GitHub](https://github.com/Xenonesis/webllm/issues) or contact the maintainer directly.