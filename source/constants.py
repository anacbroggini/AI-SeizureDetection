

# CHANNELS = [
#     'F4-C4', 
#     'F3-C3', 
#     'FT9-FT10', 
#     'FZ-CZ', 
#     'F7-T7', 
#     'FP2-F4', 
#     'T8-P8-1', 
#     'T8-P8-0', 
#     'FP1-F3', 
#     'CZ-PZ'
#     ]

CHANNELS = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZPZ', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8']
FREQUENCY_RANGES = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50),
}

DEFAULT_PATIENTS = [1, 2, 3, 4, 5, 7, 8, 9, 11, 17, 18, 19, 21, 22]