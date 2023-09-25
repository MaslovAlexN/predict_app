from flask import Flask, request, jsonify

app = Flask(__name__)

class ModelPipeline:
    def init(self, model_paths):
        self.models = {}
        for model_name, model_path in model_paths.items():
            self.models[model_name] = joblib.load(model_path)
    
    def predict_all(self, text, id_):
        predictions = {}
        for model_name in self.models:
            model = self.models[model_name]
            prediction[model_name] = model.predict(text)
        return predictions

pipeline = ModelPipeline(
        model_paths={
            'sample_model_1': 'model_1.pkl',
            'sample_model_2': 'model_2.pkl',
        }
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    text = data.get('text')
    id_ = data.get('id')
    
    if not text or not id_:
        return jsonify({"error": "Both 'text' and 'id' fields are required."}), 400
    
    predictions = pipeline.predict_all(text, id_)
    
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)