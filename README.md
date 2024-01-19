### A docker image containing all the required code (`pythia8`, `rivet`, `yoda`, `jupyterlab`) is available. Pull and run it interactively with `docker run -it alialkadhim/pythia_sbi_tune`

To run the docker you could do 

```
docker run -v $PWD:$PWD -w $PWD -p 8890:8890 -it pythia_sbi_tune
```

Then, to start a jupyter server, inside the docker container do
```
jupyter-lab --ip 0.0.0.0 --port 8890 --allow-root &
```

Then copy the url that is displayed in the terminal and paste it in your local browser. 
