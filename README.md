# capacity
Capacity of all cycles for various batteries


import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

UPPER_LIMIT = 3.44
LOWER_LIMIT = 3.28

UPPER_RANGE = 3.410
LOWER_RANGE = 3.365
DELTA_V_THRESHOLD = 0.008

login_url = "https://api.lime.ai/lime/login"
login_response = requests.post(login_url, json={"emailId":"zenenergy@lime.ai","password":"LIME_ax,K^?M5_Xvv"}, timeout=30)
login_response.raise_for_status()
token = login_response.json().get("result").get("oAuthId")

url = "https://api.lime.ai/lime/iotData"
headers = {
    "Authorization": f"{token}",
    "Content-Type": "application/json"
}

batteries = [
    {"imei": "865044073993034", "name": "Battery 1"},
    {"imei": "865044074001548", "name": "Battery 2"},
    {"imei": "865044073954416", "name": "Battery 3"},
    {"imei": "865044073953665", "name": "Battery 4"},
    {"imei": "865044073953269", "name": "Battery 5"},
    {"imei": "865044073967657", "name": "Battery 6"},
]

# Fetch data 
all_battery_data = []

for battery in batteries:
    payload = {
        "startDate": "2025-12-28",
        "endDate": "2026-02-20",
        "imei": battery["imei"]
    }
    
    print(f"Fetching data for {battery['name']} ({battery['imei']})...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        result = data["result"]
        
        if result:
            df = pd.DataFrame(result)
            df['battery_name'] = battery['name']
            df['imei'] = battery['imei']
            all_battery_data.append(df)
            print(f"  - Retrieved {len(df)} records")
        else:
            print(f"  - No data found")
    except Exception as e:
        print(f"  - Error: {e}")

# Combine all battery data
if all_battery_data:
    combined_df = pd.concat(all_battery_data, ignore_index=True)
    print(f"\nTotal combined records: {len(combined_df)}")
    print(f"Columns available: {list(combined_df.columns)}")
else:
    print("No data retrieved from any battery")
    combined_df = pd.DataFrame()

#charging events
combined_df['batCurrent'] = pd.to_numeric(combined_df['batCurrent'], errors='coerce')
charging_df = combined_df[(combined_df['batCurrent'] > 0) & (combined_df['CFETOnOffStatus'] == 1)].copy()

if len(charging_df) > 0:
    charging_df['timeStamp'] = pd.to_datetime(charging_df['timeStamp'])
    print(f"\nCharging records found: {len(charging_df)}")
    print(charging_df.head())
    charging_df.to_csv("data.csv", index=False)
    print("Charging events saved to data.csv")
else:
    print("\nNo charging events found with current filters.")
    print("Try adjusting the filtering criteria.")
    charging_df = pd.DataFrame()


def create_c2_c1_vs_cycles_graph(data):
    
    
    # If no data was passed, return an empty dictionary to prevent errors
    if len(data) == 0:
        return {}
        
    print("\nProcessing data for c2-c1 vs cycles graph...")
    df = pd.DataFrame(data)
    
    #timestamp to datetime
    df['timeStamp'] = pd.to_datetime(df['timeStamp'])
    
    #Sort by timestamp
    df = df.sort_values('timeStamp')
    
    # Get unique batteries
    batteries_in_data = df['battery_name'].unique() if 'battery_name' in df.columns else ['Unknown']
    print(f"Batteries found in data: {list(batteries_in_data)}")
    
    # Store results for each battery
    all_battery_results = {}
    
    for battery_name in batteries_in_data:
        battery_df = df[df['battery_name'] == battery_name].copy()
        
        # charging sessions (gap > 5 minutes = new session)
        battery_df['time_diff'] = battery_df['timeStamp'].diff().dt.total_seconds() / 60
        battery_df['session_id'] = (battery_df['time_diff'] > 5).cumsum()
        
        # Total cycles for this battery
        total_cycles = battery_df['session_id'].nunique()
        print(f"\n{battery_name}: Total cycles detected: {total_cycles}")
        
        # Process each session
        cycle_data = []
        
        for session_id, session_df in battery_df.groupby('session_id'):
            if len(session_df) < 2:
                continue
            
            # Sort by timestamp
            session_df = session_df.sort_values('timeStamp').copy()
            
            # Convert necessary columns to numeric
            session_df['batCurrent'] = pd.to_numeric(session_df['batCurrent'], errors='coerce').fillna(0)
            
            #avg cell voltage columns
            cell_cols = [f'cell{i}Volt' for i in range(1, 17) if f'cell{i}Volt' in session_df.columns]
            for col in cell_cols:
                session_df[col] = pd.to_numeric(session_df[col], errors='coerce').fillna(0)
            
            # Calculate avg cell voltage
            session_df['cell_voltage_avg'] = session_df[cell_cols].mean(axis=1) / 1000
            
            # start and max voltage
            start_voltage = session_df['cell_voltage_avg'].iloc[0]
            max_voltage = session_df['cell_voltage_avg'].max()
            
            # capacity calculation
            session_df['time_diff_seconds'] = session_df['timeStamp'].diff().dt.total_seconds()
            session_df['time_diff_hours'] = session_df['time_diff_seconds'] / 3600
            session_df['capacity_interval'] = session_df['batCurrent'] * session_df['time_diff_hours']
            session_df['cumulative_capacity'] = session_df['capacity_interval'].cumsum().fillna(0)
            
            voltage_data = session_df['cell_voltage_avg'].values
            capacity_data = session_df['cumulative_capacity'].values
            
            # Filter: start <= 3.28V AND max >= 3.44V
            if start_voltage <= LOWER_LIMIT and max_voltage >= UPPER_LIMIT:
                
                c1_point = None
                c2_point = None
                
                range_mask = (voltage_data >= LOWER_RANGE)
                range_indices = np.where(range_mask)[0]
                
                if len(range_indices) > 1:
                    range_voltages = voltage_data[range_indices]
                    delta_v = np.diff(range_voltages)
                    
                    if len(delta_v) > 0:
                        threshold = DELTA_V_THRESHOLD
                        valid_deltas = np.abs(delta_v) <= threshold
                        
                        # Find c1
                        c1_idx = None
                        for i in range(len(valid_deltas)):
                            if valid_deltas[i]:
                                c1_idx = range_indices[i]
                                break
                        if c1_idx is None:
                            c1_idx = range_indices[-1]
                        
                        # Find c2
                        c2_idx = None
                        below_upper_range_mask = (voltage_data >= LOWER_RANGE) & (voltage_data < UPPER_RANGE)
                        below_upper_indices = np.where(below_upper_range_mask)[0]
                        
                        if len(below_upper_indices) > 1:
                            below_upper_voltages = voltage_data[below_upper_indices]
                            delta_v_below = np.diff(below_upper_voltages)
                            
                            if len(delta_v_below) > 0:
                                invalid_deltas = np.abs(delta_v_below) > threshold
                                for i in range(len(invalid_deltas)):
                                    if invalid_deltas[i]:
                                        c2_idx = below_upper_indices[i]
                                        break
                                if c2_idx is None:
                                    c2_idx = below_upper_indices[-1]
                        
                        if c2_idx is None:
                            c2_idx = range_indices[-1]
                        
                        # Set c1 point
                        if c1_idx is not None and c1_idx < len(capacity_data):
                            c1_point = {
                                'capacity': capacity_data[int(c1_idx)],
                                'voltage': voltage_data[int(c1_idx)]
                            }
                        
                        # Set c2 point
                        if c2_idx is not None and c2_idx < len(capacity_data):
                            c2_point = {
                                'capacity': capacity_data[int(c2_idx)],
                                'voltage': voltage_data[int(c2_idx)]
                            }
                
                # Calculate c2-c1 capacity difference
                if c1_point and c2_point:
                    c2_c1_diff = c2_point['capacity'] - c1_point['capacity']
                    
                    # Get average voltage and current for this session
                    avg_voltage = session_df['cell_voltage_avg'].mean()
                    avg_current = session_df['batCurrent'].mean()
                    session_start = session_df['timeStamp'].min()
                    
                    cycle_data.append({
                        'cycle': session_id,
                        'c2_c1_capacity': c2_c1_diff,
                        'c1_capacity': c1_point['capacity'],
                        'c2_capacity': c2_point['capacity'],
                        'c1_voltage': c1_point['voltage'],
                        'c2_voltage': c2_point['voltage'],
                        'avg_voltage': avg_voltage,
                        'avg_current': avg_current,
                        'timestamp': session_start
                    })
        
        all_battery_results[battery_name] = cycle_data
        print(f"  Filtered cycles (start <={LOWER_LIMIT}V, max >={UPPER_LIMIT}V): {len(cycle_data)}")
        
    return all_battery_results


# Call the function and store the returned value
all_battery_results = create_c2_c1_vs_cycles_graph(charging_df)

# Plotting block
battery_imei_map = {b["name"]: b["imei"] for b in batteries}


fig = go.Figure()

# Add a trace for each battery to the single figure
if all_battery_results:
    for battery_name, cycle_data in all_battery_results.items():
         if len(cycle_data) > 0:
            cycles = [d['cycle'] for d in cycle_data]
            c2_c1_values = [d['c2_c1_capacity'] for d in cycle_data]
                
            # Get the IMEI for this battery from the mapping
            battery_imei = battery_imei_map.get(battery_name, 'Unknown')
            
            fig.add_trace(go.Scatter(x=cycles, y=c2_c1_values, mode='lines+markers', name=f'Battery {battery_imei}'))

    fig.update_layout(
        title="C2-C1 Capacity vs Cycles",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity Difference (c2-c1)"
    )
    fig.show()
