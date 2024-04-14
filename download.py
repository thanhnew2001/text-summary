from transformers import AutoModel, AutoTokenizer
import os

def download_model(model_name, token, save_directory):
    # # Setting up authentication token
    os.environ["HF_TOKEN"] = token

    # Create the directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Downloading the model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Saving model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer have been saved to {save_directory}")

# Replace 'your_model_name_here' with the actual model name
# Replace 'your_huggingface_token_here' with your actual token
# Replace 'path_to_save_directory' with the path where you want to save the model and tokenizer
download_model('Eugenememe/mix-en-vi-4m', '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', 'model-en-vi')
download_model('Eugenememe/mix-vi-en-1m', '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', 'model-vi-en')





