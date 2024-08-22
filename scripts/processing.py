import random
import string
import sys
import os
import time

import fiiireflyyy.logic_gates
import numpy as np
import toml
import logging
from fiiireflyyy import files as ff
from fiiireflyyy import process as fp

import pandas as pd

from scripts import data_processing as dpr

with open(os.path.join(os.getcwd(), "config/process.toml")) as f:
    config = toml.load(f)

logger = logging.getLogger("__process__")


def generate_harmonics(freq, nth, mode):
    harmonics = []
    step = freq
    if mode == 'All':
        for i in range(nth):
            harmonics.append(freq)
            freq = freq + step
    if mode == "Even":
        for i in range(nth):
            if i % 2 == 0:
                harmonics.append(freq)
                freq = freq + step
    if mode == "Odd":
        for i in range(nth):
            if i % 2 == 1:
                harmonics.append(freq)
                freq = freq + step
    return harmonics


def task_preparation():
    # ------------------------------------------------
    #             PREPARATION OF TASKS
    # ------------------------------------------------
    logger.info("Getting files")
    all_files = []
    if config["filesorter"]["enable_multiple"] is True:
        files = ff.get_all_files(config["filesorter"]["multiple"]["parent_directory"])
        for file in files:
            if all(i in file for i in config["filesorter"]["multiple"]["to_include"]) and (
                    not any(e in file for e in config["filesorter"]["multiple"]["to_exclude"])):
                all_files.append(file)
    else:
        all_files.append(config["filesorter"]["single"]["file"])
    
    logger.debug(f"Files used for processing: {all_files}")
    
    n_files = len(all_files)
    behead = config["signal"]["behead"] if config["signal"]["behead"] else 0
    example_dataframe = pd.read_csv(all_files[0], index_col=False, skiprows=behead)
    n_columns = config["signal"]["select_columns"]["number"] if config["signal"]["select_columns"]["number"] \
        else int(len([col for col in example_dataframe.columns if config["signal"]["index_col"] not in col]))
    
    # compute number of tasks needed
    subdivide = 1 if config["signal"]["subdivide"] else 0
    average = 1 if config["signal"]["average"] else 0
    interpolation = 1 if config["signal"]["interpolation"] else 0
    make_dataset = 1 if config["save"]["make_as_dataset"] else 0
    filtering = 1 if config["signal"]["filtering"]["enable"] else 0
    fft = 1 if config["signal"]["fft"] else 0
    
    file_level_tasks = behead + n_columns + subdivide
    sample_level_tasks = average + interpolation
    column_level_tasks = filtering * n_columns + fft * n_columns
    
    total_tasks = n_files * (file_level_tasks + config["signal"]["subdivide"]
                             * (sample_level_tasks + column_level_tasks))
    if make_dataset:
        total_tasks += n_files * config["signal"]["subdivide"]
    
    logger.info(f"Number of tasks : {total_tasks}")
    
    return all_files


def filename_preparation():
    # ------------------------------------------------
    #             PREPARATION OF FILENAME
    # ------------------------------------------------
    processing_basename = []
    characters = string.ascii_letters + string.digits
    
    # append filename only if filename is true
    if config["save"]["filename"]:
        processing_basename.append(config["save"]["filename"])
    else:
        if config["signal"]["select_columns"]["number"]:
            processing_basename.append(f'Sel{config["signal"]["select_columns"]["mode"].capitalize()}'
                                       f'{config["signal"]["select_columns"]["metric"].capitalize()}'
                                       f'{config["signal"]["select_columns"]["number"]}')
        if config["signal"]["subdivide"]:
            processing_basename.append(f'Sub{config["signal"]["subdivide"]}')
        if config["signal"]["filtering"]["enable"]:
            second_freq = "-" + config["signal"]["filtering"] if config["signal"]["filtering"]["type"] in ["bandstop",
                                                                                                           "bandpass"] \
                else ""
            harmonic_name = (f'H{config["signal"]["harmonics"]["type"]}{config["signal"]["harmonics"]["frequency"]}'
                             f'-{config["signal"]["harmonics"]["nth"]}' if config["signal"]["harmonics"]["enable"]
                             else "")
            processing_basename.append(
                f'O{config["signal"]["filtering"]["order"]}{config["signal"]["filtering"]["type"]}'
                f'{config["signal"]["filtering"]["first_freq"]}{second_freq}'
                f'{harmonic_name}'
            )
        if config["signal"]["fft"]:
            processing_basename.append("FFT")
        if config["signal"]["average"]:
            processing_basename.append("avg")
        if config["signal"]["interpolation"]:
            processing_basename.append(f'Ip{config["signal"]["interpolation"]}')
    
    if config["save"]["random_key"]:
        processing_basename.append(''.join(random.choice(characters) for _ in range(5)))
    if config["save"]["keyword"]:
        processing_basename.append(config["save"]["keyword"])
    if config["save"]["timestamp"]:
        processing_basename.append(time.strftime("%Y-%m-%d-%H-%M"))
    
    if not config["save"]["filename"] and not config["save"]["random_key"] and not config["save"]["keyword"] \
            and not config["save"]["timestamp"]:
        processing_basename.append("FireLearnConsole_processed")
    
    return processing_basename


def process():
    check_params()
    all_files = task_preparation()
    behead = config["signal"]["behead"] if config["signal"]["behead"] else 0
    processed_files_to_make_dataset = []
    
    processing_basename = filename_preparation()
    
    # ------------------------------------------------
    #             HARMONICS GENERATION
    # ------------------------------------------------
    logger.info(f"Harmonics generation...")
    print(processing_basename)
    harmonics = []
    if config["signal"]["harmonics"]["enable"] is True:
        harmonics = generate_harmonics(config["signal"]["harmonics"]["frequency"],
                                       config["signal"]["harmonics"]["nth"],
                                       config["signal"]["harmonics"]["type"], )
    
    logger.debug(f"Harmonics = {harmonics}")
    # ------------------------------------------------
    #             FILE PROCESSING
    # ------------------------------------------------
    logger.info(f"File processing...")
    logger.debug(f"filesorter = {config['filesorter']}")
    logger.debug(f"Row selection = {config['signal']['select_rows']}")
    logger.debug(f"Column selection = {config['signal']['select_columns']}")
    logger.debug(f"Subdivision = {config['signal']['subdivide']}")
    logger.debug(f"Filtering = {config['signal']['filtering']}")
    logger.debug(f"Filtering harmonics = {config['signal']['harmonics']}")
    logger.debug(f"FFT = {config['signal']['fft']}")
    logger.debug(f"Averaging signals = {config['signal']['average']}")
    logger.debug(f"Interpolation = {config['signal']['interpolation']}")
    
    n_file = 1
    for file in all_files:
        logger.info(f"processing file {file}")
        if behead:
            df = pd.read_csv(file, index_col=False, skiprows=behead, dtype=np.float64)
        else:
            df = pd.read_csv(file, index_col=False)
        
        if config['signal']['select_row']['enable']:
            df = df.loc[config['signal']['select_row']['start_index']:config['signal']['select_row']['end_index'], :]
        
        # select columns
        if config["signal"]["select_columns"]["number"]:
            df = dpr.top_n_electrodes(df, config["signal"]["select_columns"]["number"],
                                      config["signal"]["index_col"])
        # subdividing
        if config["signal"]["subdivide"]:
            samples = fp.equal_samples(df, config["signal"]["subdivide"])
        else:
            samples = [df, ]
        
        # processing subdivisions
        n_samples = 0
        for df_s in samples:
            df_s_fft = pd.DataFrame()
            
            # filtering
            if config["signal"]["filtering"]["enable"]:
                for ch in [col for col in df_s.columns if config["signal"]["index_col"] not in col]:
                    df_s_ch = df_s[ch]
                    if config["signal"]["filtering"]["type"] in ['highpass', 'lowpass'] \
                            and config["signal"]["filtering"]["first_freq"]:
                        df_s_ch = dpr.butter_filter(df_s_ch, order=config["signal"]["filtering"]["order"],
                                                    btype=config["signal"]["filtering"]["type"],
                                                    cut=config["signal"]["filtering"]["first_freq"])
                    elif config["signal"]["filtering"]["type"] in ['bandstop', 'bandpass'] \
                            and config["signal"]["filtering"]["first_freq"] \
                            and config["signal"]["filtering"]["second_freq"]:
                        df_s_ch = dpr.butter_filter(df_s_ch, order=config["signal"]["filtering"]["order"],
                                                    btype=config["signal"]["filtering"]["type"],
                                                    lowcut=config["signal"]["filtering"]["first_freq"],
                                                    highcut=config["signal"]["filtering"]["second_freq"])
                    
                    if config["signal"]["harmonics"]["enable"]:
                        for h in harmonics:
                            df_s_ch = dpr.butter_filter(df_s_ch,
                                                        order=config["signal"]["filtering"]["order"],
                                                        btype='bandstop',
                                                        lowcut=h - 2,
                                                        highcut=h + 2)
                    
                    df_s.loc[:, ch] = df_s_ch  # updating the dataframe for further processing
            
            # Fast Fourier Transform
            if config["signal"]["fft"]:
                for ch in [col for col in df_s.columns if config["signal"]["index_col"] not in col]:
                    df_s_ch = df_s[ch]
                    
                    # fast fourier
                    clean_fft, clean_freqs = dpr.fast_fourier(df_s_ch, config["signal"]["fft"])
                    if "Frequency [Hz]" not in df_s_fft.columns:
                        df_s_fft['Frequency [Hz]'] = clean_freqs
                    df_s_fft[ch] = clean_fft
                df_s = df_s_fft
            
            # merge signal
            if config["signal"]["average"]:
                df_s = dpr.merge_all_columns_to_mean(df_s, "Frequency [Hz]").round(3)
            
            # smoothing signal
            df_s_processed = pd.DataFrame()
            if config["signal"]["interpolation"]:
                for ch in df_s.columns:
                    df_s_processed[ch] = fp.smoothing(df_s[ch], config["signal"]["interpolation"],
                                                      'mean')
            else:
                df_s_processed = df_s
            
            # saving file
            filename_constructor = []
            filename = os.path.basename(file).split(".")[0]
            
            filename_constructor.append(filename)
            filename_constructor.append("_".join(processing_basename))
            filename_constructor.append(".csv")
            
            if not config["save"]["make_as_dataset"]:
                df_s_processed.to_csv(
                    os.path.join(config["save"]["save_under"], '_'.join(filename_constructor)),
                    index=False)
            else:
                processed_files_to_make_dataset.append((df_s_processed, file))
            n_samples += 1
        
        logger.info(f"Files processed : {n_file}/{len(all_files)}")
        n_file += 1
    
    if config["save"]["make_as_dataset"]:
        logger.info(f"Building dataset... {len(processed_files_to_make_dataset)} files to process")
        first_df = processed_files_to_make_dataset[0][0]
        dataset = pd.DataFrame(columns=[str(x) for x in range(len(first_df.values))])
        targets = pd.DataFrame(columns=['label', ])
        
        # Building dataset
        for data in processed_files_to_make_dataset:
            dataframe = data[0]
            file = data[1]
            for col in dataframe.columns:
                if "time" not in col.lower() and "frequency" not in col.lower():
                    signal = dataframe[col].values
                    dataset.loc[len(dataset)] = signal
                    for key_target, value_target in config["filesorter"]["multiple"]["targets"].items():
                        if key_target in file:
                            targets.loc[len(targets)] = value_target
        
        dataset['label'] = targets['label']
        
        dataset.to_csv(
            os.path.join(config["save"]["save_under"], '_'.join(processing_basename) + '.csv'),
            index=False)
        
        logger.info("Processing completed.")


def value_has_forbidden_character(value):
    # forbidden_characters = "<>:\"/\\|?*[]" with slashes
    forbidden_characters = "<>:\"|?*[]"
    found_forbidden = []
    for fc in forbidden_characters:
        if fc in value:
            found_forbidden.append(fc)
    
    return found_forbidden


def check_params():
    if config['filesorter']['enable_multiple'] and config['filesorter']['single']['file']:
        raise ValueError('toml: You can only chose one between Single file analysis or Multiple files.')
    
    if not config['filesorter']['enable_multiple'] and not config['filesorter']['single']['file']:
        raise ValueError('toml: You have to enable one between Single file analysis or Multiple files.')
    
    if config['filesorter']['enable_multiple'] and not config['filesorter']['multiple']['parent_directory']:
        raise ValueError('toml: You have to select a parent directory to use multiple file analysis.')
    
    if config['signal']['select_rows']['enable']:
        if config['signal']['select_rows']['start_index'] > config['signal']['select_rows']['end_index']:
            raise ValueError('toml: On row selection, le first index must be inferior to the second.')
    
    for key, value in config['filesorter']['multiple']['targets'].items():
        fcs = value_has_forbidden_character(value)
        if fcs:
            raise ValueError(f'toml: target value {value} has a forbidden character {fcs}')
    
    if config['signal']['harmonics']['enable']:
        if config['signal']['harmonics']['nth']:
            harmonic = config['signal']['harmonics']['frequency']
            nth = config['signal']['harmonics']['nth']
            frequency = config['signal']['filtering']['sampling_frequency']
            if harmonic * nth > frequency / 2:
                raise ValueError("Toml: The chosen nth harmonic is superior to half the sampling frequency."
                                 f" Please use maximum nth harmonic as nth<{int((frequency / 2) / harmonic)}")
        else:
            raise ValueError("toml: You have to fill both the harmonic frequency and the nth harmonics"
                             " using valid numbers.")
    
    if not fiiireflyyy.logic_gates.AND([config['signal']['filtering']['order'],
                                        config['signal']['filtering']['sampling_frequency'],
                                        config['signal']['filtering']['first_freq']]):
        raise ValueError('toml: You have to fill at least the filter order, sampling '
                         'frequency, and first frequency to use the filtering function.')
    
    if config["signal"]["filtering"]["enable"] and not config['signal']['filtering']['first_freq']:
        raise ValueError('toml: To filter, the first frequency is mandatory')
    
    if config['signal']['filtering']['type'] in ['bandpass', 'bandstop']:
        if not config['signal']['filtering']['first_freq'] and not config['signal']['filtering']['second_freq']:
            raise ValueError('toml: both the first and second frequency are need when using filters of type bandstop '
                             'or bandpass')
        
        if config['signal']['filtering']['first_freq'] >= config['signal']['filtering']['second_freq']:
            raise ValueError('toml: When using bandpass or bandstop filters, second freq > first freq')
        
    if not config['save']['make_as_dataset'] and not config['filesorter']['enable_multiple']:
        raise ValueError('toml: The "make dataset" option is available only if "Merge" and "Multiple files analysis" '
                         'are both True.')
