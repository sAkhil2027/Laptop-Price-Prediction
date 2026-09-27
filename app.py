from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Pre-compute unique sorted option lists from dataset
COMPANIES = sorted(df["Company"].unique())
TYPES = sorted(df["TypeName"].unique())
RAMS = sorted([int(x) for x in df["Ram"].unique()])
WEIGHTS = sorted(list(set([round(float(w), 2) for w in df["Weight"].unique()])))
CPUS = list(df["Cpu brand"].unique())
HDDS = sorted([int(x) for x in df["HDD"].unique()])
SSDS = sorted([int(x) for x in df["SSD"].unique()])
GPUS = list(df["Gpu brand"].unique())
OSS = list(df["os"].unique())
BRAND_OS_MAP = df.groupby('Company')['os'].unique().apply(list).to_dict()
TYPE_WEIGHT_MAP = {k: round(float(v), 2) for k, v in df.groupby('TypeName')['Weight'].median().to_dict().items()}

SCREEN_SIZES = [10.1, 11.3, 11.6, 12.0, 12.3, 12.5, 13.0, 13.3, 13.5, 13.9, 14.0, 14.1, 15.0, 15.4, 15.6, 17.0, 17.3, 18.4]
RESOLUTIONS = ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']

@app.route("/", methods=["GET", "POST"])
def home():
    price = None
    error_msg = None
    selected = {}

    if request.method == "POST":
        try:
            # get inputs from form
            company = request.form.get("company", COMPANIES[0])
            type_ = request.form.get("type", TYPES[0])
            ram_raw = request.form.get("ram", "8")
            weight_raw = request.form.get("weight", "auto")
            touchscreen = request.form.get("touchscreen", "No")
            ips = request.form.get("ips", "No")
            screen_size_raw = request.form.get("screen_size", "15.6")
            resolution = request.form.get("resolution", "1920x1080")
            cpu = request.form.get("cpu", CPUS[0])
            hdd_raw = request.form.get("hdd", "0")
            ssd_raw = request.form.get("ssd", "256")
            gpu = request.form.get("gpu", GPUS[0])

            # Resolve 'Not Sure / Standard' auto options
            ram = int(ram_raw) if ram_raw not in ["auto", ""] else 8
            if weight_raw in ["auto", "", "Not Sure"]:
                weight = TYPE_WEIGHT_MAP.get(type_, 1.8)
            else:
                weight = float(weight_raw)

            screen_size = float(screen_size_raw) if screen_size_raw not in ["auto", ""] else 15.6
            if resolution in ["auto", "Not Sure"]:
                resolution = "1920x1080"

            hdd = int(hdd_raw) if hdd_raw not in ["auto", "", "Not Sure"] else 0
            ssd = int(ssd_raw) if ssd_raw not in ["auto", "", "Not Sure"] else 256

            # Validate OS against selected brand
            valid_oss_for_brand = BRAND_OS_MAP.get(company, OSS)
            submitted_os = request.form.get("os")
            os = submitted_os if submitted_os in valid_oss_for_brand else valid_oss_for_brand[0]

            selected = {
                "company": company,
                "type": type_,
                "ram": ram if ram_raw != "auto" else "auto",
                "weight": weight_raw,
                "touchscreen": touchscreen,
                "ips": ips,
                "screen_size": screen_size if screen_size_raw != "auto" else "auto",
                "resolution": request.form.get("resolution", "1920x1080"),
                "cpu": cpu,
                "hdd": hdd_raw,
                "ssd": ssd_raw,
                "gpu": gpu,
                "os": os
            }

            # Map touchscreen & IPS flags
            touch_val = 1 if touchscreen == "Yes" else 0
            ips_val = 1 if ips == "Yes" else 0

            # Calculate PPI
            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2 + Y_res**2) ** 0.5) / screen_size

            query = np.array([
                company, type_, ram, weight, touch_val,
                ips_val, ppi, cpu, hdd, ssd, gpu, os
            ]).reshape(1, 12)

            price = int(np.exp(pipe.predict(query)[0]))
        except Exception as e:
            error_msg = f"An error occurred while predicting: {str(e)}"

    return render_template(
        "index.html",
        companies=COMPANIES,
        types=TYPES,
        rams=RAMS,
        weights=WEIGHTS,
        screen_sizes=SCREEN_SIZES,
        resolutions=RESOLUTIONS,
        cpus=CPUS,
        hdds=HDDS,
        ssds=SSDS,
        gpus=GPUS,
        oss=OSS,
        brand_os_map=BRAND_OS_MAP,
        price=price,
        error_msg=error_msg,
        selected=selected
    )
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

