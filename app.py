from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("house_prices_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            area = float(request.form["area"])
            bedrooms = int(request.form["bedrooms"])
            age = int(request.form["age"])

            input_data = np.array([[area, bedrooms, age]])
            result = model.predict(input_data)
            prediction = f"{result[0]:.2f} Lakhs"
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
