# Active Learning for Precise Event Spotting (ALPES)

This repository contains a research project conducted as part of [CP3209 - Undergraduate Research Project in Computing (UROP)](https://www.comp.nus.edu.sg/programmes/ug/project/urop/).

The project investigates the use of data-efficient machine learning methods (primarily active learning) to improve precise event spotting (PES). The focus is on reducing the amount of labeled data required for training while preserving model performance in PES tasks.

## Overview

### Precise Event Spotting (PES)

Precise event spotting (PES) is a computer vision task that, given a video, aims to determine _what_ event occurred and exactly _when_ it happened. It can be viewed as a fine-grained version of temporal action localization (TAL). While TAL typically involves identifying when an event occurs on the order of seconds, PES aims to localize the event on a much finer timescale—often at the level of individual frames.

### Active Learning (AL)

[Active learning (AL)](<https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>) is a machine learning approach that reduces the amount of labeled data needed for training. Instead of labeling all data upfront, the model selects the most informative or uncertain samples and queries an oracle (typically a human annotator) for their labels. By focusing labeling effort on these critical samples, AL can achieve strong performance with fewer annotations as compared to traditional training methods.
