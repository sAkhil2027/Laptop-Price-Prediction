from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def home():
    price = None

    if request.method == "POST":
        # get inputs from form
        company = request.form["company"]
        type_ = request.form["type"]
        ram = int(request.form["ram"])
        weight = float(request.form["weight"])
        touchscreen = request.form["touchscreen"]
        ips = request.form["ips"]
        screen_size = float(request.form["screen_size"])
        resolution = request.form["resolution"]
        cpu = request.form["cpu"]
        hdd = int(request.form["hdd"])
        ssd = int(request.form["ssd"])
        gpu = request.form["gpu"]
        os = request.form["os"]

        # same logic as Streamlit
        touchscreen = 1 if touchscreen == "Yes" else 0
        ips = 1 if ips == "Yes" else 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

        query = np.array([
            company, type_, ram, weight, touchscreen,
            ips, ppi, cpu, hdd, ssd, gpu, os
        ]).reshape(1, 12)

        price = int(np.exp(pipe.predict(query)[0]))

    return render_template(
        "index.html",
        companies=df["Company"].unique(),
        types=df["TypeName"].unique(),
        cpus=df["Cpu brand"].unique(),
        gpus=df["Gpu brand"].unique(),
        oss=df["os"].unique(),
        price=price
    )

if __name__ == "__main__":
    app.run(debug=True)
