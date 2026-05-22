"# Voice Recognition" 
First record the samples by running *record_samples* file, write the name then the samples count, by clicking enter every time it is ready to record for 4 sec.
Second, after recording the samples it will create automatically a file called *wav_files* it contain all the samples of each person, now you can go to the next step which is running file *extract_features*.
Then go to file *train_knn* in which gives you the accuracy of your samples after converting.
lastly, run *mic_test* file and you will have 4 sec to speak then it will detect the speaker with a confidience ratio.
Additions(*convert_all* to convert the records from phone *m4a* to *wav* ready to use on laptop, *test_model* file is to test by applying a wav record and see if he can detic without talking live)
