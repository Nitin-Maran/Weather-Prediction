from flask import Flask, request, render_template_string
from datetime import datetime
from main import predict_weather

app = Flask(__name__)

template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Weather Predictor</title>
</head>
<body>
    <h2>Predict Weather</h2>
    <form method="POST">
        Date & Time: <input type="datetime-local" name="datetime" required><br><br>
        <input type="submit" value="Predict Weather">
    </form>

    {% if prediction %}
        <h3>Predicted Temperature: {{ prediction.temperature }} °C</h3>
        <h3>Predicted Humidity: {{ prediction.humidity }}</h3>
        <h3>Predicted Weather: {{ prediction.summary }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        datetime_str = request.form['datetime']
        dt = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M')
        month = dt.month
        day = dt.day
        hour = dt.hour

        temp, humidity, summary = predict_weather(month, day, hour)
        prediction = {
            'temperature': temp,
            'humidity': humidity,
            'summary': summary
        }

    return render_template_string(template, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
