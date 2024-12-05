"""
Created on December 2024

@author: Angelos
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus

# Set the working directory
# working_directory = 'D:/MaritimeGCH/Greece/webapp'
# os.chdir(working_directory)

# Read input data from CSV files
def getParameters(demand_file, op_cost_file, emissions_factor_file, fuel_cost_file, co2_cap_file, ets_price_file, fuel_consumption_file):
    params = {
        "years": range(2020, 2051),  # Planning horizon
        "ship_types": ["C", "T", "B", "G", "O"],  # container, tanker, bulk, cargo, other
        "engine_types": ["ME-C", "ME-GI", "ME-LGI"],
        "init_capacity_fleet": (
            pd.read_csv("init_capacity_fleet.csv", index_col="ship_type")["capacity"].to_dict()
        ),
        "minim_capacity_fleet": (
            pd.read_csv("minim_capacity_fleet.csv", index_col="ship_type")["limit"].to_dict()
        ),
        "fleet_age": (
            pd.read_csv("init_age.csv", index_col="ship_type")["avr_age"].to_dict()
        ),
        "demand_shipping": (
            pd.read_csv(demand_file)
            .set_index(["year", "ship_type"])["demand"]
            .to_dict()
        ),
        "investment_cost": (
            pd.read_csv("investment_cost.csv", index_col="ship_type")["cost"].to_dict()
        ),
        "op_cost": (
            pd.read_csv(op_cost_file, index_col="ship_type")["cost"].to_dict()
        ),
        "fuel_cost": (
            pd.read_csv(fuel_cost_file, index_col="fuel_type")["cost"].to_dict()
        ),
        "co2_cap": pd.read_csv(co2_cap_file, index_col="year")["cap"].to_dict(),
        "ets_price": pd.read_csv(ets_price_file, index_col="year")["price"].to_dict(),
        "emissions_factor": (
            pd.read_csv(emissions_factor_file, index_col="fuel_type")["factor"].to_dict()
        ),
        "prod_capacity": (
            pd.read_csv("prod_capacity.csv")
            .set_index(["year", "ship_type"])["capacity"]
            .to_dict()
        ),
        "lifetime": (
            pd.read_csv("lifetime.csv", index_col="ship_type")["years"].to_dict()
        ),
        "fuel_consumption": (
            pd.read_csv(fuel_consumption_file)
            .set_index(["ship_type", "fuel_type", "engine_type", "year"])["consumption"]
            .to_dict()
        ),
        "fuel_avail": (
            pd.read_csv("fuel_avail_med.csv")
            .set_index(["fuel_type", "year"])["availability"]
            .to_dict()
        ),
        "cap": pd.read_csv("cap.csv", index_col="ship_type")["capacity"].to_dict(),
        "CII_desired": (
            pd.read_csv("CII_desired.csv", index_col="ship_type")["CII"].to_dict()
        ),
    }
    params["fuel_types"] = list(params["fuel_cost"].keys())
    return params


def createAndSolveModel(params):
    # Create the LP problem instance
    model = LpProblem(name="MaritimeGCHgr", sense=LpMinimize)

    # Decision Variables
    new_ship = {
        (y, s): LpVariable(name=f"new_ship_{y}_{s}", lowBound=0, cat="Integer")
        for y in params["years"]
        for s in params["ship_types"]
    }
    stock_ship = {
        (y, s): LpVariable(name=f"stock_ship_{y}_{s}", lowBound=0, cat="Integer")
        for y in params["years"]
        for s in params["ship_types"]
    }
    fuel_demand = {
        (y, f): LpVariable(name=f"fuel_demand_{y}_{f}", lowBound=0)
        for y in params["years"]
        for f in params["fuel_types"]
    }
    co2_emissions = {
        y: LpVariable(name=f"co2_emissions_{y}", lowBound=0) for y in params["years"]
    }
    excess_emissions = {
        y: LpVariable(name=f"excess_emissions_{y}", lowBound=0)
        for y in params["years"]
    }

    ##### Objective Function: Minimize total cost
    model += lpSum(
        new_ship[y, s] * params["investment_cost"].get(s, 0)
        + stock_ship[y, s] * params["op_cost"].get(s, 0)
        + fuel_demand[y, f] * params["fuel_cost"].get(f, 0)
        for y in params["years"]
        for s in params["ship_types"]
        for f in params["fuel_types"]
    ) + lpSum(
        excess_emissions[y] * params["ets_price"].get(y, 0)
        for y in params["years"]
    )

    ##### Constraints
    # Fleet Capacity Constraint
    for y in params["years"]:
        for s in params["ship_types"]:
            model += (
                stock_ship[y, s] * params["cap"].get(s, 0) >= params["demand_shipping"].get((y, s), 0)
            )

    # Ship Production Constraint
    for y in params["years"]:
        for s in params["ship_types"]:
            model += new_ship[y, s] <= params["prod_capacity"].get((y, s), 0)

    # Fleet Stock Update Constraint
    for y in params["years"]:
        for s in params["ship_types"]:
            if y == 2020:
                model += stock_ship[y, s] == params["init_capacity_fleet"].get(s, 0)
            else:
                retired_ships = lpSum(
                    new_ship[max(2020, y - params["lifetime"].get(s, 1) + 1 - params["fleet_age"].get(s, 0)), s]
                    for y_prev in range(max(2020, y - params["lifetime"].get(s, 1) + 1), y)
                )
                model += stock_ship[y, s] == stock_ship[y-1, s] + new_ship[y, s] - retired_ships

    # Fuel Demand Calculation
    for y in params["years"]:
        for f in params["fuel_types"]:
            fuel_demand_value = lpSum(
                stock_ship[y, s]
                * params["fuel_consumption"].get((s, f, eng, y), 0) * 1e-2
                for s in params["ship_types"]
                for eng in params["engine_types"]
            )
            model += fuel_demand[y, f] == fuel_demand_value

    # Emissions Constraint
    for y in params["years"]:
        model += co2_emissions[y] == lpSum(
            fuel_demand[y, f] * params["emissions_factor"].get(f, 0) * 10e-6
            for f in params["fuel_types"]
        )

    # ETS Emissions Cap Constraint
    for y in params["years"]:
        model += co2_emissions[y] <= params["co2_cap"].get(y, 0) + excess_emissions[y]

    # Solve the model
    model.solve()

    return model


# Extract Results
def extract_results(model, params):
    variables = model.variables()
    years = list(params['years'])
    
    # Initialize results dictionary
    results = {
        'Year': years,
        'CO2_Emissions': [0 for _ in years],
        'Total_Cost': [model.objective.value() for _ in years],
        'Investment_Cost': [0 for _ in years],
        'Operational_Cost': [0 for _ in years],
        'Fuel_Cost': [0 for _ in years],
        'excess_emissions': [0 for _ in years],
        'ets_penalty': [0 for _ in years],
    }
    
    # Initialize ship type results
    for s in params['ship_types']:
        results[f'New_Ships_{s}'] = [0 for _ in years]
        results[f'Stock_Ships_{s}'] = [0 for _ in years]
    
    # Initialize fuel type results
    for f in params['fuel_types']:
        results[f'Fuel_Demand_{f}'] = [0 for _ in years]

    # Loop through the variables and fill in the results
    for v in variables:
        name_parts = v.name.split('_')
        if name_parts[0] == 'co2':
            year = int(name_parts[2])
            year_index = years.index(year)
            results['CO2_Emissions'][year_index] = v.varValue
        elif name_parts[0] == 'new':
            year = int(name_parts[2])
            year_index = years.index(year)
            ship_type = name_parts[3]
            results[f'New_Ships_{ship_type}'][year_index] = v.varValue
            results['Investment_Cost'][year_index] += v.varValue * params['investment_cost'].get(ship_type, 0)
        elif name_parts[0] == 'stock':
            year = int(name_parts[2])
            year_index = years.index(year)
            ship_type = name_parts[3]
            results[f'Stock_Ships_{ship_type}'][year_index] = v.varValue
            results['Operational_Cost'][year_index] += v.varValue * params['op_cost'].get(ship_type, 0)
        elif name_parts[0] == 'fuel':
            year = int(name_parts[2])
            year_index = years.index(year)
            fuel_type = name_parts[3]
            results[f'Fuel_Demand_{fuel_type}'][year_index] = v.varValue
            results['Fuel_Cost'][year_index] += v.varValue * params['fuel_cost'].get(fuel_type, 0)
        elif name_parts[0] == 'excess':
            year = int(name_parts[2])
            year_index = years.index(year)
            results['excess_emissions'][year_index] = v.varValue

    # Calculate the penalized (ETS-taxed) excess CO2 emissions
    for i, year in enumerate(years):
        results['ets_penalty'][i] = results['excess_emissions'][i] * params['ets_price'].get(year, 0)

    return pd.DataFrame(results)



# Streamlit App
st.title("MaritimeGCH: The Investment Decision Support Tool of the Global Climate Hub, optimizing fleets, to meet techno-economic, environmental and regulatory goals")

st.markdown("""
**The MaritimeGCH** is an advanced optimization model using dynamic linear programming to minimize the total cost of fleet operations until 2050. 
It incorporates a comprehensive set of parameters, including ship types, engine types, fuel types, initial fleet capacities, demand for shipping services, 
investment and operational costs, fuel costs, CO2 emissions and associated EU taxes based on the Emissions Trading System (ETS), new-built ships capacities 
and their age and lifetimes, fuel consumption rates, and fuel availability. MaritimeGCH optimizes new ship acquisitions, existing fleet management, fuel consumption, 
and CO2 emissions while adhering to operational and environmental constraints, including the recent ETS and Carbon Intensity Indicator (CII) compliance indicator.  

** Ship types:** Containers (C), Tankers (T), Bulk (B), Cargo (C), Other (O).
** Fuels:** Oil, RefPO, LNG, LPG, MeOH, NH3, H2

** Objective: ** Minimize the total cost over the planning horizon (2020-2050)

** Constraints:** The total stock of ships each year must be sufficient to meet the demand for shipping services

The number of new ships built each year is limited by the production capacity

The stock of ships of each type in a given year is the sum of new ships built and surviving ships from previous years, depending on their age and lifetime

The fuel demand is derived from the operational needs of the ships, which however, cannot exceed the available amount of each fuel type

The total CO2 emissions are calculated based on fuel consumption. If the CO2 emissions exceed the ETS cap (threshold), then this will be penalized (ETS penalty).

The Carbon Intensity Indicator (CII) must comply with the EU regulation thresholds.

**DOI:** [10.13140/RG.2.2.35892.87680](https://doi.org/10.13140/RG.2.2.35892.87680)  
**Available at:** [GitHub Repository](https://github.com/Alamanos11/MaritimeGCH)
""")

st.sidebar.header("Select a scenario based on the following input parameters. Case study: Fleet under Greek flag!")

demand_file = st.sidebar.selectbox("Select a shipping demand scenario, based on the Shared Socioeconomic Pathways (SSP1, 2, or 5)", ['demand_shippingSSP1.csv', 'demand_shippingSSP2.csv', 'demand_shippingSSP5.csv'])
op_cost_file = st.sidebar.selectbox("Select a shipping technology (engine optimization, hull cleaning, port call, propulsion system, route optimization, or their combination). These technologies reduce emissions, but come at a cost", ['op_cost_engin_opt.csv', 'op_cost_hull.csv', 'op_cost_port_call.csv', 'op_cost_propul.csv', 'op_cost_route_opt.csv', 'op_cost_comb.csv'])
emissions_factor_file = st.sidebar.selectbox("For the shipping technology selected above, chose here its respective emission reduction potential", ['emissions_factor_engin_opt.csv', 'emissions_factor_hull.csv', 'emissions_factor_port_call.csv', 'emissions_factor_propul.csv', 'emissions_factor_route_opt.csv', 'emissions_factor_comb.csv'])
fuel_cost_file = st.sidebar.selectbox("Select a fuel costs market situation (low, medium, or high costs)", ['fuel_cost_low.csv', 'fuel_cost_med.csv', 'fuel_cost_high.csv'])
co2_cap_file = st.sidebar.selectbox("Select a CO2 emissions Cap (threshold) based on which the ETS will penalize the exceeding emissions (no penalty, pessimistic, medium, or optimistic emissions target", ['co2_cap_no.csv', 'co2_cap_pess.csv', 'co2_cap_med.csv', 'co2_cap_opt.csv'])
ets_price_file = st.sidebar.selectbox("Select ETS Price for the excess emissions penalty (no penalty, a moderate, or a strict one)", ['ets_price_no.csv', 'ets_price_mod.csv', 'ets_price_strict.csv'])
fuel_consumption_file = st.sidebar.selectbox("Select a pace for the transition to greener fuels by 2050 (slow, medium, fast)", ['fuel_consumption_slow.csv', 'fuel_consumption_med.csv', 'fuel_consumption_fast.csv'])

if st.sidebar.button("Run Model"):
    st.write("### Running Optimization...and see Results:")
    
    # Pass all the required files to getParameters() function
    parameters = getParameters(
        demand_file=demand_file, 
        op_cost_file=op_cost_file,
        emissions_factor_file=emissions_factor_file,
        fuel_cost_file=fuel_cost_file,
        co2_cap_file=co2_cap_file,
        ets_price_file=ets_price_file,
        fuel_consumption_file=fuel_consumption_file
    )
    
    # Solve the model
    model = createAndSolveModel(parameters)
    
    # Extract results
    results = extract_results(model, parameters)

    # Create plots
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))

    # Plot Stock Ships
    bottom = np.zeros(len(results['Year']))
    for s in parameters['ship_types']:
        axes[0, 0].bar(results['Year'], results[f'Stock_Ships_{s}'], bottom=bottom, label=s)
        bottom += results[f'Stock_Ships_{s}']
    axes[0, 0].set_title('Stock Ships [number]')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Number of Stock Ships')
    axes[0, 0].legend()

    # Plot New Ships
    bottom = np.zeros(len(results['Year']))
    for s in parameters['ship_types']:
        axes[0, 1].bar(results['Year'], results[f'New_Ships_{s}'], bottom=bottom, label=s)
        bottom += results[f'New_Ships_{s}']
    axes[0, 1].set_title('New Ships [number]')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Number of New Ships')
    axes[0, 1].legend()

    # Plot Investment Costs
    axes[1, 0].plot(results['Year'], results['Investment_Cost'], marker='o')
    axes[1, 0].set_title('Investment Costs [million Euros]')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Investment Costs')

    # Plot Operational Costs
    axes[1, 1].plot(results['Year'], results['Operational_Cost'], marker='o')
    axes[1, 1].set_title('Operational Costs [million Euros]')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Operational Costs')

    # Plot Fuel Demand
    bottom = np.zeros(len(results['Year']))
    for f in parameters['fuel_types']:
        axes[2, 0].bar(results['Year'], results[f'Fuel_Demand_{f}'], bottom=bottom, label=f)
        bottom += results[f'Fuel_Demand_{f}']
    axes[2, 0].set_title('Fuel Demand [tonnes]')
    axes[2, 0].set_xlabel('Year')
    axes[2, 0].set_ylabel('Fuel Demand')
    axes[2, 0].legend()

    # Plot Fuel Costs
    axes[2, 1].plot(results['Year'], results['Fuel_Cost'], marker='o')
    axes[2, 1].set_title('Fuel Costs [million Euros]')
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].set_ylabel('Fuel Costs')

    # Plot CO2 Emissions and Cap
    axes[3, 0].plot(results['Year'], results['CO2_Emissions'], label='Total CO2 Emissions')
    axes[3, 0].plot(results['Year'], parameters['co2_cap'].values(), label='CO2 Cap', linestyle='--')
    axes[3, 0].set_title('CO2 Emissions and Cap [million tonnes]')
    axes[3, 0].set_xlabel('Year')
    axes[3, 0].set_ylabel('CO2 Emissions')
    axes[3, 0].legend()

    # Plot ETS Penalty
    axes[3, 1].plot(results['Year'], results['ets_penalty'], marker='o')
    axes[3, 1].set_title('ETS Penalty [million Euros]')
    axes[3, 1].set_xlabel('Year')
    axes[3, 1].set_ylabel('Penalty Costs')

    # Adjust layout
    plt.tight_layout(pad=4.0)

    # Show the plots in Streamlit
    st.pyplot(fig)



#### To run the web app, go to Anaconda prompt and paste: 
#### streamlit run d:\maritimegch\greece\webapp.py