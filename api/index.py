from dotenv import load_dotenv
load_dotenv()          # ← this loads .env file automatically
import os
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI
import cloudinary
import cloudinary.uploader
import cloudinary.api

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["UPLOAD_FOLDER"] = "static/uploads"  # still used locally if needed

db = SQLAlchemy(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ────────────────────────────────────────────────
#  Cloudinary setup – runs once when app starts
# ────────────────────────────────────────────────
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Database Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result = db.Column(db.String(2000))
    image_path = db.Column(db.String(500))  # longer for Cloudinary URLs
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# AI Client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_API_KEY")  # better to use env var than hardcode
)

# ────────────────────────────────────────────────
# MODEL FUNCTIONS
# ────────────────────────────────────────────────

def text_model(prompt_text):
    try:
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2.5",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=800,
            temperature=0.85
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Text model error: {str(e)}"


def image_url_model(image_url, prompt_text):
    try:
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2.5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            max_tokens=800,
            temperature=0.85
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Image URL model error: {str(e)}"


def image_upload_model(file, prompt_text):
    try:
        base64_image = base64.b64encode(file.read()).decode('utf-8')
        response = client.chat.completions.create(
            model="moonshotai/Kimi-K2.5",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=800,
            temperature=0.85
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Image upload model error: {str(e)}"


# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_static(path):
    print(f"Requested path: /{path}")
    full_path = os.path.join("static", "index.html" if path == "" else path)
    print(f"Trying to serve: {os.path.abspath(full_path)}")
    print(f"File exists? {os.path.exists(full_path)}")

    if path == "":
        return send_from_directory("static", "index.html")
    return send_from_directory("static", path)

@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode")

    if not mode:
        return jsonify({"result": "Error: mode field is missing"}), 400

    # Prompts with instructions for rich formatting
    prompts = {
        "describe": "Describe this image in full detail with emojis to highlight features and **bold** key elements. Be engaging and comprehensive.",
        "caption": "Write a detailed caption for this image, using emojis for emphasis and **bold** important words. Make it vivid and full.",
        "objects": "List all objects in this image in detail, with emojis next to each and **bold** the main ones. Provide full descriptions.",
        "explain": "Explain the scene in this image thoroughly, using emojis to illustrate points and **bold** key concepts. Be detailed and engaging."
    }

    result = ""
    image_path = ""  # will store Cloudinary URL or empty

    try:
        if mode == "text":
            prompt_text = request.form.get("prompt", "").strip()
            if prompt_text:
                enhanced_prompt = f"{prompt_text} Respond in full detail with emojis and **bold** text for emphasis."
                result = text_model(enhanced_prompt)
            else:
                result = "Please enter a prompt."

        elif mode == "url":
            image_url = request.form.get("image_url", "").strip()
            prompt_type = request.form.get("prompt_type", "describe")
            instruction = prompts.get(prompt_type, prompts["describe"])
            if image_url:
                result = image_url_model(image_url, instruction)
            else:
                result = "Please provide a valid image URL."

        elif mode == "upload":
            if "image" not in request.files or not request.files["image"].filename:
                result = "No image file selected."
            else:
                file = request.files["image"]
                prompt_type = request.form.get("prompt_type", "describe")
                instruction = prompts.get(prompt_type, prompts["describe"])

                # ────────────────────────────────────────────────
                # Upload to Cloudinary
                # ────────────────────────────────────────────────
                try:
                    upload_result = cloudinary.uploader.upload(
                        file,
                        resource_type="image",
                        folder="ai-vision-uploads",  # optional
                        use_filename=True,
                        unique_filename=False
                    )
                    image_url = upload_result["secure_url"]
                    print(f"Cloudinary upload success: {image_url}")

                    # Send URL to vision model
                    result = image_url_model(image_url, instruction)

                    # Save Cloudinary URL for history
                    image_path = image_url

                except Exception as e:
                    result = f"Cloudinary upload failed: {str(e)}"
                    print(result)

        else:
            result = "Invalid mode."

    except Exception as e:
        result = f"Error during processing: {str(e)}"

    # Save to database
    record = Prediction(result=result, image_path=image_path)
    db.session.add(record)
    db.session.commit()

    return jsonify({"result": result})


@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.id.desc()).all()

    history_data = [
        {
            "id": r.id,
            "result": r.result,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "image": r.image_path if r.image_path else ""   # full Cloudinary URL
        }
        for r in records
    ]

    return jsonify(history_data)


@app.route("/delete/<int:id>", methods=["DELETE"])
def delete_record(id):
    record = Prediction.query.get(id)
    if record:
        db.session.delete(record)
        db.session.commit()
        return jsonify({"status": "deleted"})
    return jsonify({"status": "not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
