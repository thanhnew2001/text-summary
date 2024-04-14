from flask import Flask, request, jsonify
from utils import summarize

app = Flask(__name__)


@app.route("/summarize", methods=["POST"])
def summarize_text():
    try:
        # Get source text from the POST request
        src_text = request.json.get("src_text", "")
        if not src_text:
            return jsonify({"error": "No source text provided"}), 400

        # Call the summarize function
        summary_en, summary_vi = summarize(src_text)

        # Return the summaries
        return jsonify({"summary_en": summary_en, "summary_vi": summary_vi})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app with debugging enabled
