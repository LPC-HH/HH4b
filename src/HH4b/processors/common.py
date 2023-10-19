# TODO: Check if this is correct for available datasets

# Golden JSON
LUMI = {  # in pb^-1
    "2018": 59830.0,
    # 5.0707 + 3.0063
    "2022": 7971.4,
    # "2022EE": 26337.0,
    # 2022EE minus E run
    "2022EE": 20665.0,
    "2023": 28072.0,
}

# TODO: split over sources of JECs
jecs = {
    "JES": "JES_jes",
    "JER": "JER",
}

jmsr = {
    "JMS": "JMS",
    "JMR": "JMR",
}

jec_shifts = []
for key in jecs:
    for shift in ["up", "down"]:
        jec_shifts.append(f"{key}_{shift}")

jmsr_shifts = []
for key in jmsr:
    for shift in ["up", "down"]:
        jmsr_shifts.append(f"{key}_{shift}")
