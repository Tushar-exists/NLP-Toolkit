# Unified NLP Toolkit ✨

Welcome to the **Unified NLP Toolkit**! This project provides a web-based interface for performing various Natural Language Processing (NLP) tasks, including text summarization, multi-language translation (specifically English to Indian local languages), and document-based question answering. Built with Gradio and Hugging Face Transformers, it offers a user-friendly way to interact with powerful NLP models.

## Table of Contents 📚

* [Features](#features)

* [Why This Toolkit Over Google Translate?](#why-this-toolkit-over-google-translate)

* [Technologies Used](#technologies-used)

* [Setup and Installation](#setup-and-installation)

  * [Prerequisites](#prerequisites)

  * [Clone the Repository](#clone-the-repository)

  * [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)

  * [Install Dependencies](#install-dependencies)

  * [Run the Application](#run-the-application)

* [How to Use](#how-to-use)

  * [Text Summarizer](#text-summarizer)

  * [Multilanguage Translator](#multilanguage-translator)

  * [Document Q&A](#document-qa)

* [Contributing](#contributing)

* [License](#license)

* [Credits](#credits)

## Features 🚀

This toolkit integrates three core NLP functionalities:

### 1. Text Summarizer 📝

* **Model:** `t5-base`

* **Functionality:** Condenses long pieces of text into concise summaries, extracting the most important information.

### 2. Multilanguage Translator 🌍

* **Model:** `facebook/nllb-200-distilled-600M`

* **Functionality:** Translates English text into a wide range of Indian local languages using FLORES-200 codes. This model offers a good balance between broad language coverage and performance.

### 3. Document Q&A ❓

* **Model:** `distilbert-base-uncased-distilled-squad`

* **Functionality:** Allows users to upload a `.txt` file and ask questions about its content, receiving direct answers extracted from the document.

## Why This Toolkit Over Google Translate? 🤔

While Google Translate is a powerful and fast general-purpose translation service, our project offers several key advantages, especially for specific use cases:

1. **Customization and Control:** ⚙️

   * **Tailored Models:** Unlike proprietary, black-box models, we use open-source models that can be fine-tuned for specific domains (e.g., legal, medical, niche dialects) to achieve higher accuracy and nuance for particular needs.

   * **Adaptability:** As new, better open-source models are released, or as data requirements change, we can easily update the backend of our toolkit.

2. **Integration and Extensibility:** 🔗

   * This project is a **unified toolkit**, combining summarization, translation, and Q&A in one interface, streamlining workflows.

   * Built on open-source libraries, it's highly extensible. New NLP tasks, custom pre-processing steps, or integrations with other systems can be easily added.

3. **Cost-Effectiveness (for High Volume/Specific Deployments):** 💰

   * Running open-source models on your own infrastructure can be more cost-effective in the long run for large-scale or continuous use cases, as you're paying for compute, not per-character API calls.

4. **Privacy & Data Control:** 🔒

   * For sensitive data, using an open-source model running locally or on a private server means you have complete control over your data, without sending it to a third-party service.

In essence, our project provides the **flexibility, control, and extensibility** needed for specialized applications, data privacy requirements, and integrated workflows that a generic, black-box API cannot offer.

## Technologies Used 💻

* **Python:** The core programming language.

* **PyTorch:** Deep learning framework for model execution.

* **Hugging Face Transformers:** Provides pre-trained NLP models and pipelines.

* **Gradio:** For creating the interactive web-based user interface.

* **JSON:** For storing language mapping data.

## Setup and Installation ⬇️

Follow these steps to get the Unified NLP Toolkit running on your local machine.

### Prerequisites ✅

* **Python 3.9 - 3.12** (Python 3.13 might have compatibility issues with some libraries like `sentencepiece` at the moment).

* **Git** (for cloning the repository).

* **Visual C++ Build Tools for Python** (essential for compiling some Python packages like `sentencepiece` on Windows). If you don't have them, download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop development with C++" and "Python development" workloads.

* **VS Code** (recommended IDE).

### Clone the Repository 📥

First, clone the project repository to your local machine:

```bash
git clone [https://github.com/Tushar-exists/NLP-Toolkit](https://github.com/Tushar-exists/NLP-Toolkit)
cd NLP-Toolkit
```

### Create and Activate Virtual Environment 🐍

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1
```

*(For macOS/Linux, use `source venv/bin/activate`)*

### Install Dependencies 📦

With your virtual environment activated, install the required Python packages:

```bash
pip install torch gradio transformers sentencepiece
```

*(Note: This step will download large models and libraries, so it may take some time depending on your internet speed and system specifications.)*

### Run the Application ▶️

Once all dependencies are installed, you can run the Gradio application:

```bash
python PT.py
```

The application will start, and a local URL (e.g., `http://127.0.0.1:7860`) will be displayed in your terminal. Open this URL in your web browser to access the toolkit.

## How to Use 📖

### Text Summarizer ✍️

1. Navigate to the "Text Summarizer" tab.

2. Paste your desired text into the "Input" text box.

3. Click "Summarize".

4. The concise summary will appear in the "Summary" output box.

### Multilanguage Translator 🗣️

1. Navigate to the "Multilanguage Translator" tab.

2. Paste your English text into the "English Text" input box.

3. Select your desired Indian language from the "Select Indian Language" dropdown.

4. Click "Translate".

5. The translated text will appear in the "Translated Text" output box.

### Document Q&A 📄

1. Navigate to the "Document Q&A" tab.

2. Click "Upload Text File" to upload a `.txt` document.

3. Type your question about the document's content into the "Your Question" text box.

4. Click "Get Answer".

5. The answer extracted from the document will appear in the "Answer" output box.

## Contributing 🤝

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please open an issue or submit a pull request.

## License 📜

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

## Credits 🙏

* Developed by Tushar
  * Connect on [LinkedIn](https://www.linkedin.com/in/tusharkbhushan) <img src="https://img.icons8.com/color/20/000000/linkedin--v1.png" alt="LinkedIn icon"/>
  * Reach out via [Email](mailto:tushar.kr.bh@gmail.com) <img src="https://img.icons8.com/color/20/000000/gmail-new.png" alt="Gmail icon"/>

* Powered by Hugging Face Transformers, PyTorch, and Gradio.
