# FireLearn Console v0.0.1

FireLearn Console (FLC) is a console-driven application developed by Willy Lutz that 
aims to provide an easy way to analyze multiple numerical data, such as electrical 
recordings. It uses the `fiiireflyyy` python library by the same author, to provide 
machine learning and other processing and analytical tools, without the need to write a single 
line of code. The only thing the user will be required to do is to modify the 
`.toml` configuration files to their needs. And of course the command line to start the application.

Everything the user needs to know will be explained thoroughly in this document.

## Table of contents




# Installation and usage

No prior knowledge of console and command-line arguments is needed, although preferred. 
Everything will be explicit in this document.
However, you need to have `Python 3.10` or more installed on your system 
([https://www.python.org/downloads/](https://www.python.org/downloads/)), 
and `git` ([https://git-scm.com/downloads](https://git-scm.com/downloads)).

You can check their correct installaction by using respectively in the terminal
```console
python3 --version
```
and
```console
git --version
```
It should print th installed version, such as `Python 3.10.12` and `git version 2.34.1`.

## Installation
### Clone via git
Next is to clone the project locally on your system. To do so, open a terminal on the desired location to 
download the project.
Open a terminal window at this location, then enter the following command
```console
git clone https://github.com/WillyLutz/FireLearnConsole.git
```

wait for a bit, and you should have a following result as following
![clone](data/help/git-clone-repo.png)
### Set up a virtual environment
It is then recommended to set up a virtual environment. If you wish not, go to the next section.
To set up the environment, use the following command.
```console
python3 -m venv /your/path/MyDesiredLocation/FireLearnConsole/venv
```
Change the path indicated here to your actual path. Any name can replace `venv`, as you see fit. However,
if you do so keep in mind you will have to adapt the other command accordingly.

For the moment, an automatic management of the virtual environment has not been implemented. So you will need to
activate the environment by using 
```console
source venv/bin/activate
```
while in the project directory. a `(venv)` should appear at the beginning of the line.

To deactivate it, simply use
```console
deactivate
```

Once activated, use
```console
pip install -r requirements.txt
```
It will then proceed to install the required dependencies for the project.

### Make the project executable
The last step is to make the main file executable. To do so, enter the directory `FireLearnConsole`.

Then use the command
```console
chmod +x firelearn.py
```
to grant executable permission to the file.
The project is now ready to use !

### Update the project
If you already have the project set up and want the latest version, you can easily update it. 
Go inside the FireLearnConsole directory, then use the command
```console
git pull origin master
```
It will be updated to the latest changes in the GitHub.
Then, activate your virtual environment and update the used libraries by using 
```console 
pip install -r requirements.txt
```
while in the project directory.
## How to use

To use the application, open a terminal and go to the project directory. Then 
[activate your virtual environment](#set-up-a-virtual-environment). 

You can now execute the application in your terminal by typing, as base synthax,
```console
python3 firelearn.py #ADD SOME ARGUMENTS
```
The arguments after will depend on what the user intends to do (processing, learning, analysis...)

The list of available arguments is [here](#available-arguments)
Each argument enables functionalities of the application, each one of them controlled independently by its configuration
file under a `.toml` extension. Those files contain the different parameters that can (and should) 
be tuned to the user needs.

### Available arguments
* `-p` Enables the processing, controlled by `processing.toml`
* `-l` Enables the learning, controlled by `learning.toml`
* `-c` Enables the confusion analysis, controlled by `confusion.toml`
* `-i` Enables the feature importances analysis, controlled by `feature_importances.toml`
* `-pca` Enables the PCA analysis, controlled by `pca.toml`
* `-plot` Enables the dataset plotting analysis, controlled by `simple_plot.toml`

If multiple arguments are in the command line, they will be executed in the order presented above.

### The .toml configuration files
Each one of the configuration files in `FireLearnConsole/config/` allow the user to fine tune the behavior of 
the application for each feature independently.
They are of the format `.toml`, which has the following architecture:

```toml
[category1]
var1 = '...'
var2 = '...'

   # comments on subcategory1
   [category1.subCategory1]
   sub_var1 = '...'
   sub_var2 = '...'
   
   [category1.subCategory2]
   sub_var1 = '...'
   sub_var2 = '...' # comments on sub_var2

[category2]
var1 = '...'
```
The indentation is not relevant, but is maintained for better readability. 
 You may encounter comments in the form `# lorem ipsum` that give 
specifications on the use of a certain category or variable.

You can edit them in any text file editor, given you respect the notation.

> <img alt="red_warning" height="30" src="data/help/red_warning.png" width="30"/>
> 
> You must not change the names of the variables, only their values.

> <img alt="red_warning" height="30" src="data/help/yellow_warning.png" width="30"/>
> 
> When modifying the variables value, make sure that they stay of their initial types (integers stay integers, 
> string stay string, lists stay lists...).


If needed, a small guide on how toml works is [available here](#https://toml.io/en/).

More specific use of our configuration files will be found in their relevant section.
# Walkthrough

## Processing
Configuration file : `processing.toml`
### Example: directory structure
For this document we will proceed considering this recommended directory structure : 
```
DATA (most parent common directory)
│
└───-DRUG
│   └───T=0MIN
│   │   └───NI
│   │   │   |
│   │   │   file.txt
│   │   │   file1_Analog.csv
│   │   │   file2_Analog.csv
│   │   │   file3_Analog.csv
│   │   │   file4_Analog.csv
│   │   └───INF
│   │       |
│   │       [...]
│   └───T=24H
│       └───[...]
└───SOMEDRUG
│   └───[...]
└───SOMEOTHERDRUG
    └───[...]
```
### Sorting multiple files
This functionality aims at looking for and using multiple files under a common parent 
directory, no matter how distant it is. 

enables it by setting 
```toml
[filesorter]
enable_multiple = true # disable by setting 'false'
```

#### Selecting parent directory
```toml
[filesorter.multiple]
# Absolute path required
parent_directory="/my/path/to/parent/dir/DATA/-DRUG/T=24H"
```
The selected directory must be a parent of all the files you want to process.
For a multiple files processing, you __must__ set `enable_multiple = true`.

For instance, using [this directory structure](#example-directory-structure), all the files that
are children to the most parent directory (here `/DATA`) are subject to be comprised in the processing.

To specify which files to include or exclude of the processing, 
refer to [the include/exclude option](#include-and-exclude-files-for-the-processing).

#### Include and exclude files for the processing
```toml
[filesorter.multiple]
to_include = ['Analog.csv', ]
to_exclude = ['TTX', ]
```
With this functionality, you can specify which file to include or exclude from the selection.
Both the inclusion and exclusion works by looking at the content of the absolute paths of the files 
(e.g. `H:\Electrical activity\DATA\-DRUG\T=0MIN\INF\Electrode Raw Data1_Analog.csv`). 

As such, the inclusion uses the AND logic operator : 
**The file is included if ALL the `to_include` specifications are present in the absolute path**.
On the other hand, the exclusion uses the OR logic operation :
**The file is excluded if ANY of the `to_exclude` specifications are present in the absolute path**.
Combining both gates, a file will be included for the processing if its absolute path 
**contains all the `to include` specifications and none of the `to exclude` specifications**.

<img alt="include_exclude.png" height="374" src="data/help/include_exclude.png" width="531"/>

Those `to include` and `to exclude` specifications are case-sensitive, so `Analog.csv` is different 
from `analog.csv`.

To add a specification, type it in the corresponding entry, then either click on the `+` button, or press 
the `Return` key. To remove a specification, type it in the corresponding entry, then either click on the 
`-` button or press `Ctrl-BackSpace` combination.

E.g. : As per the previous figure, only the files that contains  "Analog.csv" **AND** "T0" **AND DO NOT CONTAIN** "TTX"
in their absolute paths will be used for further processing.
#### Indicating targets for learning
```toml
[filesorter.multiple.targets]
'NI' = 'Mock'
'INF' = 'Infected'
```
It is possible to make an entry correspond with a label for future processing, based on its path. 

To do so, you will need to create associations of 'key' and 'value', using the synthax
```toml
'key' = 'value'
```
indicate as the `key` a sequence of characters to find in the path of the file. Then indicate as the `value` 
the corresponding label. When creating the resulting files for the analysis, a label (`value`) will be assigned to 
the data using the `key` provided. 

E.g. : in the previous toml section, the label `Mock` will be assigned to the data if `NI` is comprised in the absolute
path of the file used. The label `Infected` will be assigned to the data if `INF` is comprised in the absolute
path of the file used. 

> <img height="30" width="30" src="data/help/yellow_warning.png" />
> 
> Be it for including or excluding, the sequence will be searched in the absolute path and not
> only on the file name. 

> <img height="30" width="30" src="data/help/red_warning.png">
> 
> The said sequences does not search for "separated sequences" and does not recognise if the sequence
> is a word in itself. It will only look at the sequence character-wise. As such when choosing the 
> sequences you want to include or exclude, be aware of what can be in your absolute paths.
> 
> > E.g.: Your project is in a folder named 'HIV PROJECT', and further away in the children folders
> > you name the different recording conditions such as 'NI' (not infected), 'HIV' (infected by HIV)
> > BUT to select the files you specify in To exclude 'HIV', **all of your files will be excluded since
> > 'HIV' as a sequence is also present in the project folder 'HIV PROJECT' and not only as a 'condition'**.
> > 
> > To remediate to such issue, it is possible to look for the sequence '/HIV/' (use your operating system path 
> > separator) instead to ensure that we only look
> > at the folder named 'HIV' and not 'HIV PROJECT'.

#### Single file analysis
```toml
[filesorter.single]
# Absolute path required
file=""
```
In order to process a single file, provide a path to `file`. 
> <img height="30" width="30" src="data/help/yellow_warning.png" />
> 
>You can not have the single file analysis enabled and
>the multiple files analysis enabled at the same time.

#### Beheading
```toml
[signal]
# putting value to 0 disables the parameter
behead = 6
```
Beheads the n first lines of the csv file (to use if there are metadata on the first lines 
of the csv file, for instance).
Be aware that after this step there must not be anything apart from the data and a row of headers, in the data file.
#### Selecting columns
```toml
[signal]
index_col = "TimeStamp [µs]"

[signal.select_columns]
mode = "max" # select columns by their maximum metric
metric = "std"  # Metric used to select columns
number = 35  # number of columns to select, based on mode and metric
```
Allow to select the columns (electrodes, in case of MEA recordings) with a `mode` `metric` combination.
Any column that contains `TimeStamp [µs]` (case-sensitive) in its header will be ignored.
#### Recordings down sampling
```toml
[signal]
subdivide = 30
```
This functionality will divide row-wisely every file in `n` selected pieces of equal lengths.
e.g. In our walkthrough example, we use 1 minute long recordings.
Specifying a subdivision at `30` implies that the recordings will be divided in 30 pieces of 2 seconds.

> <img src="data/help/red_warning.png" width="30" height="30">
>
> If the [make resulting files as dataset](#post-processing) function is not used, be aware that each file
selected during the [selection process](#sorting-multiple-files) will generate an equal number of different
based on the [subdivision](#recordings-down-sampling).

#### Filtering
```toml
[signal.filtering]
enable = true
order = 3
sampling_frequency = 10000  # in Hz
type = 'highpass' # among 'highpass', 'lowpass', 'bandstop', 'bandpass'
first_freq = 50 # in Hz
second_freq = 0 # in Hz, only if 'bandstop' or 'bandpass'
```

Allow the user to apply a Butterworth filter to the data.

For the use of `lowpass` and `highpass` filters, the first frequency `first-freq` corresponds to the cut
frequency, and the second frequency `second_freq` must be left emptied. Both are expressed in Hertz.
For the use of `bandstop` and `bandpass` filters, the second frequency `f2 (Hz)` corresponds to the high-cut
frequency and must be specified.

```toml
[signal.harmonics]
enable = true
order = 3
type = 'all' # among 'all', 'even', 'odd'
frequency = 50  # in Hz
nth = 35
```
In order to filter harmonics, the user can choose to filter only the `Even` harmonics (2nd, 4th, 6th...),
or the `Odd` harmonics (1st, 3rd, 5th...) or `All` (1st, 2nd, 3rd, 4th...) up till the `nth` harmonic.

E.g. with a harmonic frequency of 100 Hz, filtering the odd harmonics up to the 10th will filter the 
following frequencies : 
_100 Hz(1st), 300 Hz(3rd), 500 Hz(5th), 700 Hz(7th), 900 Hz(9th)_


#### Fast Fourier Transform
```toml
[signal]
fft = 10000  # in Hz
```
Applies a Fast Fourier Transform to the data (post-filtering, if the filtering is enabled). 
The sampling rate must be specified.
#### Interpolation
```toml
[signal]
interpolation = 300
```
Interpolates the signal down to `n` final values. `n` must be inferior to the number of data point.
Uses one-dimensional linear interpolation.
#### Averaging columns
```toml
[signal]
average = true
```
#### Resulting datasets
```toml
[save]
make_as_dataset = true  # or false
```

> <img src="data/help/red_warning.png" width="30 height=30">
> 
>This feature overwrites the saving of the intermediate files (e.g. those created 
> from the [subdivision](#recordings-down-sampling) functionality) and save only
> a final file 'DATASET' csv file.

> <img src="data/help/yellow_warning.png" width="30 height=30">
>
>This feature is only available if the [columns averaging](#averaging-electrodes) is enabled. 

Enabling this feature will merge all the processed files issued from the 
[signal averaging](#averaging-electrodes) such as each merged signal results in one row in the dataset.


#### Post-processing
```toml
[save]
random_key = true
timestamp = true
keyword = ""
filename = ""
# Absolute path required
save_under = "/home/wlutz/PycharmProjects/FireLearnConsole/output"
```

The `random_key` variable allows to add a random key as combination of 6 alphanumerical characters to the 
resulting filenames.

The `timestamp` variable adds a timestamp of format `year-month-day-hour-minute` (at the time the file is being 
processed).

The `keyword` variable adds a specified keyword to the resulting filenames.

The `filename` variable allows the user to set the filename (not the path).

> <img src="data/help/yellow_warning.png" width="30" height="30">
>
> These customisations are added at the end of the filename. If `filename` is not specified, an
> automatic one will be chosen based on the processing procedures used.


The `save_under` allow to specify the directory where the resulting files will be saved.


## Learning

## Analysis

# Project information
## Support

## Authors and acknowledgement

## Licence

## Project status

