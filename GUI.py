from logging import warning
from json import (load as jsonload, dump as jsondump)
from os import path
import PySimpleGUI as sg

CONFIG_FILEPATH = "configurations.json"
# "Map" from the config dictionary keys to the window's element keys
DEFAULT_CONFIG = {
    "price_complete": {"layer": 1, "past_history": 168, "dropout": 2,
                       "epochs": 300, "batch_size": 128},
    "remainder_complete": {"layer": 2, "past_history": 72, "dropout": 5,
                           "epochs": 300, "batch_size": 64},
    "price_day": {"layer": 1, "past_history": 168, "dropout": 2,
                  "epochs": 300, "batch_size": 128},
    "remainder_day": {"layer": 2, "past_history": 72, "dropout": 3,
                      "epochs": 300, "batch_size": 64}}

CONFIG_TO_ELEM = {("price_complete", "layer"): 'PC_LAYER',
                  ("price_complete", "past_history"): 'PC_PAST_HISTORY',
                  ("price_complete", "dropout"): 'PC_DROPOUT',
                  ("price_complete", "batch_size"): 'PC_BATCH_SIZE',
                  ("remainder_complete", "layer"): 'RC_LAYER',
                  ("remainder_complete",
                   "past_history"): 'RC_PAST_HISTORY',
                  ("remainder_complete", "dropout"): 'RC_DROPOUT',
                  ("remainder_complete", "batch_size"): 'RC_BATCH_SIZE',
                  ("price_day", "layer"): 'PD_LAYER',
                  ("price_day", "past_history"): 'PD_PAST_HISTORY',
                  ("price_day", "dropout"): 'PD_DROPOUT',
                  ("price_day", "batch_size"): 'PD_BATCH_SIZE',
                  ("remainder_day", "layer"): 'RD_LAYER',
                  ("remainder_day", "past_history"): 'RD_PAST_HISTORY',
                  ("remainder_day", "dropout"): 'RD_DROPOUT',
                  ("remainder_day", "batch_size"): 'RD_BATCH_SIZE'}


def load_config():
    try:
        with open(CONFIG_FILEPATH, 'r') as f:
            config = jsonload(f)
    except Exception as e:
        sg.popup_quick_message(f'exception {e}',
                               'No config file found... will create one for you in the programs directory',
                               keep_on_top=True, background_color='red',
                               text_color='white')
        config = DEFAULT_CONFIG
        save_config(CONFIG_FILEPATH, config, None)
    return config


def save_config(config, new_config):
    if new_config:  # if there are stuff specified by another window, fill in those new config values
        for key in CONFIG_TO_ELEM:
            try:
                config[key[0]][key[1]] = new_config[CONFIG_TO_ELEM[key]]
            except Exception as e:
                print(
                    f'Problem updating config from window values. Key = {key}')

    with open(CONFIG_FILEPATH, 'w') as f:
        jsondump(config, f, indent=4)

    sg.popup('Settings saved successfully!')


def create_config_window(config):
    sg.theme('LightGrey6')


    descriptions = ["Additional layers", "Input length",
                    "dropout strength", "Batch size (Training)"]

    headers = ["Parameter",'Price complete ', 'Remainder complete', 'Price day',
               'Remainder day']
    types = ["PC", "RC", "PD", "RD"]
    variables = ["LAYER", "PAST_HISTORY", "DROPOUT", "BATCH_SIZE"]
    possible_values = [[i for i in range(0, 10)],
                       [i for i in range(1, 500)],
                       [i for i in range(0, 10)],
                       [pow(2, i) for i in range(1, 10)]]
    first_col_size = max([len(description) for description in descriptions])
    header_row=[]
    for i in range(len(headers)):
        elem=sg.Text(headers[i],pad=(15,0)) if i>0 else sg.Text(headers[i],size=(first_col_size,1))
        header_row+=[elem]
    col=[header_row]
    for i in range(len(descriptions)):
        layer_row=[sg.Text(descriptions[i],size=(first_col_size,1))]+ [sg.Spin(possible_values[0],size=(len(headers[t+1]),1),key="{}_{}".format(types[t],variables[i]) )for t in range(len(types))]
        col+=[layer_row]
    layout=col
    layout += [[sg.Button('Save'), sg.Button('Exit')]]

    window = sg.Window('Net configurations', layout, keep_on_top=True,
                       finalize=True,auto_size_text=True)

    for key in CONFIG_TO_ELEM:  # update window with the new_config read from config file
        try:
            window[CONFIG_TO_ELEM[key]].update(
                value=config[key[0]][key[1]])
        except Exception as e:
            print(
                f'Problem updating PySimpleGUI window from config. Key = {key}')

    return window


def create_main_window(settings):
    sg.theme('LightGrey6')
    layout = [

        [sg.T("TRAIN MODELS"),
         sg.CB("Price    ", default=False, key='train_complete'),
         sg.CB("Remainder", default=False, key='train_remainder'),sg.B('Change Net Configuration')],
        [sg.T("PREDICT:         "),
         sg.CB("Price     ", default=True, key='predict_complete'),
         sg.CB("remainder", default=False, key='predict_remainder'),
         sg.CB("decomposed (will override remainder)", default=True,
               key='predict_decomposed'),],
         [sg.T("",size=(len("PREDICT:    "),1)),sg.CB("SARIMA", default=True, key='predict_arima'),
         sg.CB("naive (persistence model for Price)", default=True,
               key='predict_naive_lagged'),
         sg.CB("naive (0-Model for Remainder)", default=True,
               key='predict_naive_0')],
        [sg.T("DAY MODELS"),
         sg.CB("use daymodels for Prediciton", default=False,
               key='predict_with_day'),
         sg.CB("Train All daymodels", default=False,
               key='train_day_of_week')],
        [sg.T(
            "specific forecast at timestep (will disable mass predictions):"),
            sg.Spin([i for i in range(-1, 169)], initial_value=-1,
                    key="test_pred_start_hour")],
        [sg.Button(button_text="START", focus=True), sg.Cancel()]]
    return sg.Window('Configuration Powerprice Prediction',
                     layout)


def main():
    window = None
    config = load_config()
    cancelled = False
    while True:  # Event Loop
        if window is None:
            window = create_main_window(config)
        event, values = window.read()
        if event in (None, 'Exit', 'Cancel'):
            cancelled = True
            break

        if event == "START":
            break

        if event == 'Change Net Configuration':
            event, new_config = create_config_window(config).read(
                close=True)
            if event == 'Save':
                window.close()
                window = None
                save_config(config, new_config)

    window.close()
    if cancelled:
        return None
    else:
        needed_plots = 0
        for key in ["predict_complete", "predict_naive_lagged",
                    "predict_naive_0", "predict_arima"]:
            if values[key]:
                needed_plots += 1
        if values["predict_decomposed"] or values["predict_remainder"]:
            needed_plots += 1
        return values, needed_plots


if __name__ == "__main__":
    warning(
        "Directly accessing the GUI module is only intended for debugging purposes. Power price prediction will not start. Execute \"prognose.py\" for a power price prediction.")
    main()
