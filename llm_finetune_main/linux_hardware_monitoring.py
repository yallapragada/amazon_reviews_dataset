import time
import csv
import psutil
import pynvml

# Initialize NVML
pynvml.nvmlInit()
gpu_count = pynvml.nvmlDeviceGetCount()

# Open a CSV file to write the data
with open("system_usage.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow([
        "timestamp", "cpu_percent",
        "mem_total", "mem_used", "mem_free",
        "gpu_index", "gpu_name", "gpu_utilization",
        "gpu_memory_total", "gpu_memory_used", "gpu_memory_free",
        "gpu_temperature", "gpu_power_draw"
    ])

    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()  # mem.total, mem.used, mem.free

        # Loop through all GPUs
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Some names might be returned as bytes, so decode if necessary
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode("utf-8")
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)  # in milliwatts

            writer.writerow([
                timestamp, cpu_percent,
                mem.total, mem.used, mem.free,
                i, gpu_name, util.gpu,
                mem_info.total, mem_info.used, mem_info.free,
                temp, power
            ])
        csvfile.flush()  # ensure data is written immediately
        time.sleep(1)
