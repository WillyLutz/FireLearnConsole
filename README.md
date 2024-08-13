# FireLearn Console v0.0.1

FireLearn Console (FLC) is a console-driven application developed by Willy Lutz that 
aims to provide an easy way to analyze multiple numerical data, such as electrical 
recordings. It uses the `fiiireflyyy` python library by the same author, to provide 
machine learning and other processing and analytical tools, without the need to write a single 
line of code. The only thing the user will be required to do is to modify the 
`.toml` configuration files to their needs. And of course the command line to start the application.

Everything the user needs to know will be explained thoroughly in this document.

## Table of contents
1. [Installation and usage](#installation-and-usage)   
   1. [Installation](#installation)
      2. [Install git](#install-via-git)
   2. [How to use](#how-to-use)
   3. [Available tools](#available-tools)
2. [Walkthrough](#walkthrough)
   1. [Processing](#processing)
      1. [Example: directory structure](#example-directory-structure)
      2. [Sorting multiple files](#sorting-multiple-files)
      3. [Selecting parent directory](#selecting-parent-directory)
      4. [Include and exclude files for the processing](#include-and-exclude-files-for-the-processing)
      5. [Indicating targets for learning](#indicating-targets-for-learning)
      6. [Single file analysis](#single-file-analysis)
      7. [Using raw MEA recordings](#using-raw-mea-recordings)
      8. [Selecting columns](#selecting-columns)
      9. [Recordings down sampling](#recordings-down-sampling)
      10. [Filtering](#filtering)
      11. [Fast Fourier Transform](#fast-fourier-transform)
      12. [Interpolation](#interpolation)
      13. [Averaging columns](#averaging-columns)
      14. [Resulting datasets](#resulting-datasets)
      15. [Post-processing](#post-processing)
      16. [Miscellaneous](#miscellaneous)
   2. [Learning](#learning)
   3. [Analysis](#analysis)
3. [Project information](#project-information)
   1. [Support](#support)
   2. [Authors and Acknowledgement](#authors-and-acknowledgement)
   3. [Licence](#licence)
   4. [Project status](#project-status)



# Installation and usage

No prior knowledge of console and command-line arguments is needed, although preferred. 
Everything will be explicit in this document.


## Installation
### Install via git
It is however needed to have `git` installed on your computer. You can get it on the official site 
[here](https://git-scm.com/downloads). Once downloaded, install it on your system.

Once installed, open a terminal. You can check the successful installation of git by writing
```console
git --version
```

It should print th installed version, such as `git version 2.34.1`.

Next is to clone the project locally on your system. To do so, open a terminal on the desired location to 
download the project.
Open a terminal window at this location, then enter the following command
```console
git clone https://github.com/WillyLutz/FireLearnConsole.git
```

wait for a bit, and you should have a following result as following
![clone](data/help/git-clone-repo.png)

The last step is to make the main file executable. To do so, enter the directory `FireLearnConsole` either by 
opening a new terminal at the location or by using the following command on the previously used terminal.
```console
cd FireLearnConsole/
```

Then use the command
```console
chmod +x firelearn.py
```
to grant executable permission to the file.
The project is now ready to use !

### Update the project
If you already have the project set up, you can easily update it. Go inside the FireLearnConsole directory,
then use the command
```console
git pull origin master
```
## How to use

## Available tools

# Walkthrough

## Processing
### Example: directory structure
### Sorting multiple files
#### Selecting parent directory
#### Include and exclude files for the processing
#### Indicating targets for learning
#### Single file analysis
#### Using raw MEA recordings
#### Selecting columns
#### Recordings down sampling
#### Filtering
#### Fast Fourier Transform
#### Interpolation
#### Averaging columns
#### Resulting datasets
#### Post-processing
#### Miscellaneous

## Learning

## Analysis

# Project information
## Support

## Authors and acknowledgement

## Licence

## Project status

