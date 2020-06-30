# Udacity_project_files_for_review
Collection of files that need to be reviewed

This repository contains:
- person_detect_1.py - which uses the Network class, used in earlier lessons of this nanodegree
- person_detect_2.py -  which uses the PersonDetect class, which was suggested for this project
- queue_job.sh - which contains the job-submission script
- submit_command.sh - which contains the command to submit the python script


## Problem

When running both of the scripts following error occurs:

```
Could not run Inference:  The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
results/
results/manufacturing/
results/manufacturing/cpu/
results/manufacturing/cpu/output_video.mp4
stderr.log
```

I suppose that something with in queue_job.sh or submit_command.sh must be faulty, but I reviewed it several times and can't find an error.
