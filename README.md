# SoundAPI

## API

### Deployment

The API is currently deployed on [Modal](https://modal.com/) through webhooks. If you want to test the code in *dev*, run
```
modal serve api.py
```
This command will spawn an ephemeral server that lasts until you shut the script down. Notice that it will still use the Modal computing services (i.e. you will incur in billing depending on usage). 

On the other hand, if you wish to (re-)deploy the API run 
```
modal deploy api.py
```
Under the hood Modal uses (Docker) images. The specific build is detailed in `api.py`, before the endpoints are defined. Notice that the directory containing the models, i.e. `Models/` is copied into the image at creation. Therefore, if we add a new model, we need to re-deploy the entire app. This could be ameliorated in future versions. 

Finally, the app currently supports only CPU. It is trivial to extend it to use GPUs (including T4, A10G and A100) by simply changing a line of code. Specifically, replacing the end of the `api.py` file with 
```
@stub.function(image=image, gpu="any")
@stub.asgi_app()
def fastapi_app():
    return app
```
In fact, if we use "any", it will typically use T4s. Replacing "any" with a specific GPU model, e.g. "A10G", will enforce this change. The downside is that using GPUs is much more costly and, at this point, is probably not really worth it. Ideally, one could make this a choice for the user. 

The URLs for the app are:

**DEV**
```
https://gemmoai--sound-api-fastapi-app-dev.modal.run
```

**PROD**
```
https://gemmoai--sound-api-fastapi-app.modal.run
```
Remember that the *dev* will only be active *after* running `modal serve api.py` on your local machine. 

### Token, Quotas, Authorisation and Authentication

In order to access the API endpoints and models, you need to validate your request. That is, you need to provide a token as part of the header of your API call that has to satisfy a number of criteria:

1. The token must be valid, i.e. it exists and is in our database; 
2. The number of uses associated with the token should not have exceeded the pre-allocated quota;
3. The token shall grant access to the endpoint which is being called; 
4. The token shall grant access to the model (or, in fact, any other possible parameter) which the user wants to use. 

If any of these conditions is *not* met, then some form of 400 error will be raised from the API. 

All of these aspects are managed by our Django admin panel, which is accessible [here](http://ec2-34-250-132-250.eu-west-1.compute.amazonaws.com/admin/). The credentials can be found in 1Password, under `TokenAuthService - Admin`. Once logged in, we can manage the entire token ecosystem. 

In particular, under `Token` we can create new tokens and/or change the permissions associated to each of these. By simply clicking on the `Key`, we can see the user it is associated to, if it is active and which permissions it has. In practice, under `Permissions` we shall provide a list of keywords that consitute the permission we are going to check at run time. These should include the name of the endpoints, the name of the models, and other "sensitive" data. 

Notice that, the `Key` appearing in the admin panel will need to be passed to the header of the API request as the parameter `Authorization` and, to the key, the "Bearer " string needs to be prepended. For instance, we should use `Bearer 12d4XXXXX`. 

When creating a new token, remember to also set a quota. If you don't do so, you will be returned a 500

Instead, under `Request historys`, for each call to the API we get all the details. These include the Machine Learning results, but also other info (such as date, endpoint, HTTP method, device, etc.). This database can be parsed using SQL. 

Finally, under `Request limit configs` we can find, and possibly modify, the quota associated to each token. Notice that when quota is reached all successive requests to the API will not go through. 

All the details of the calls are stored in a PostgreSQL database deployed on AWS RDS (*Relational Database Service*). One can access it from the command line using 
```
psql -h token-auth-service.cgh7bclyuzzr.eu-west-1.rds.amazonaws.com -U postgres -W
```
Notice that you will be prompted to insert a password. This can be found on 1Password, under `RDS - token-auth-service`. One can then run SQL queries on our database to fetch information. Another more graphical way to interact with the database is to install [pgAdmin](https://www.pgadmin.org/download/pgadmin-4-apt/) from which it should be easier to deal with the DB. 

### Backend

The backend is provided by [FastAPI](https://fastapi.tiangolo.com/), a Pythonic library for building APIs. 

### Docs

Thanks to FastAPI auto-generated docs are readily accessible at the `/docs` endpoint of the URL, both in *prod* and in *dev*. For example, in the case of the production code, one could access them [here](https://gemmoai--sound-api-fastapi-app.modal.run/docs). The docs are auto-generated so we can save a lot of time by just being precise when writing the Python code. One has to use Pydantic's `BaseModel` to detail the formatting of the JSON output. Moreover, one can define some examples so as to auto-fill examples. Finally (and this is very important), it is possible to try out the API with the FastAPI frontend! 

### Endpoints
- **/classify**: used to classify the audio source in one or more audio files (up to a maximum of 16 files). Files are always processed in batch. There is a hard cut-off on the length of the audio files, at the moment fixed at 5 seconds. Content past 5s is not even loaded into memory but simply discarded. 

- **/window**: used to classify the audio content of *one* audio file by subdividing it into non-overlapping windows of length specified by the user. That is, if the user uploads a file of 10s and specifies a window of 3s, then the audio will be split into four intervals, namely (0, 3), (3, 6), (6, 9), (9, 10). These windows will then be stacked one onto the other (possibly addding silence = zero padding at the end of the last one) to create a batch. 

- **/detect**: this endpoint is used to detect audio sounds in a (possibly long) audio sample. A call to it will return the most likely content of a very very short window (around 0.01 seconds). Notice that the model underpinning this endpoint is different, and is based on, among other things, an attention module. 

## AWS

Most of the data (including models, datasets, clients' data and so on) are backed-up on Gemmo's AWS account S3. It is useful to interact with it directly through the AWS CLI. However, since you might have multiple AWS accounts (e.g. the old Beautifeye), you might have a crash with credentials for different accounts. In order to avoid that proceeds as follows (courtesy of ChatGPT): 

1. Activate your venv, `source venv/bin/activate`
2. pip-install the CLI, `pip install awscli`
3. Insert the various keys after typing `aws configure --profile sound-api`
4. Use these credentials with `aws --profile sound-api <COMMAND>`

The organisation of the bucket related to the SoundAPI is as follows. First, this bucket is called `gemmoai-soundapi`. The structure is

```
gemmoai-soundapi/
├── data/
│   ├── collected/
|   │   ├── audio/
|   |   │   ├── <FILE_1>.wav
|   |   │   ├── <FILE_2>.mp3
|   |   │   └── ...
|   │   └── labels/
│   └── datasets/
|   │   ├── <DATASET_1>
|   |   │   ├── Audio
|   |   |   │   ├── <FILE_1>.wav
|   |   |   │   ├── <FILE_2>.mp3
|   |   |   │   └── ...
|   |   │   ├── index_to_label.yaml
|   |   │   └── labels.csv
├── models/
│   ├── <MODEL_1_NAME>/
|   |   ├── <WEIGHTS_1>.pt
|   |   ├── metadata.yaml
|   |   └── index_to_label.yaml
│   ├── <MODEL_2_NAME>/
|   |   ├── <WEIGHTS_2>.pt
|   |   ├── metadata.yaml
|   |   └── index_to_label.yaml
│   └── ...

```

## Client's Report

In order to create a report for a client testing the API proceed as follows. Go to `Clients/` and create a new directory with the name of the client (let's call it "MyClient" for demonstration purposes). Inside `Clients/MyClient` create yet another directory called `Audio` where you will store the audio files in whatever (admissible!) format. Hence the structure will be the following
```
Clients ¬
        |
        MyClient ¬
                 |
                 Audio ¬
                       |
                       file1.mp3
                       |
                       file2.wav
                       |
                       ...
```
Once things are organised this way, launch `report.py`. It accepts four parameters, namely:
- *model*: the name of the model to be used, e.g. *Generic* or *BuildingSite*
- *endpoint*: the name of the endpoint the client wants to see tested, e.g. *classify*
- *client*: the name of the client, i.e. the name of the directory you just created, e.g. *MyClient*
- *n-tags*: the number of tags to be returned 

An example would be
```
python report.py --model Generic --endpoint classify --client MyClient --n-tags 3
```
This will process all files in the corresponding `Audio` directory and create a report, in the form of an Excel file. This fill will be stored in `Clients\<CLIENT_NAME>`. 


## Inference

### Running Inference 
To run inference:

```
python inference.py --dataset_path <PATH> --model_name <MODEL> --batch_size <BATCH> --n_tags <N_TAGS>
```

The flags stand for:

- `dataset_path`: str, default "Data". Path to where the audio files to be processed are stored. In fact, notice that the audio files need to be placed inside `<PATH>/Audio`. 
- `model_name`: str, default "Wavegram". Name of the model to be used for inference. This is the name of the directory inside `Models` where all relevant files are stored. See below for further explanation. 
- `batch_size`: int, default 32. Batch size for inference. 
- `n_tags`: int, default 3. Maximum number of tags to be assigned to each audio file. That is, only the `<N_TAGS>` most probable classes for that audio are returned. 

### Inference Preparation 
1. Place audio files in `Data/Audio`. The parent directory, i.e. `Data`, can be changed to any other, provided one specifies it when calling the script using the flag `--dataset_path <PATH_TO_DIRECTORY>`. However, audio files need to be contained in a directory called `Audio` inside of this. 
2. Make sure you have the model, metadata and file associating indices with labels in place. That is: 
    - Inside the `Models` directory you should have a directory with the name of your model, e.g. `Wavegram`. This is arbitrary, of course, but will need to be passed to the script via the flag `--model_name <MODEL_NAME>`. 
    - Inside the directory of the specific model, i.e. `Models/<MODEL_NAME>` there should be i) the model checkpoint, 2) `metadata.yaml`, 3) `index_to_label.yaml`. The structure of `metadata.yaml` is as follows:

    ```
    module: <NAME_OF_NETWORK_MODULE>
    weights: <NAME_OF_WEIGHTS>
    parameters:
        <PARAMETER_1>: <VALUE>
        ...
    ```
    where `<NAME_OF_NETWORK_MODULE>` indicates the name of the Python module (without `.py`, and without the path) containing the architecture. This needs to be placed in `Code/Networks`. As an example, if the architecture is in `Code/Networks/wavegram_logmel_cnn14.py`, then the metadata file should contain `module: wavegram_logmel_cnn14`. 

    Weights should be the name of the file containing the checkpoint, including the extension, and without the path. This needs to be placed in the `Models/<MODEL_NAME>` directory. For instance, `weights: Wavegram_Logmel_Cnn14_mAP=0.439.pth`. 

    Finally, one should specify the parameters that are needed when loading the module. For instance, if `Code/Networks/my_model.py` has the form 

    ```
    class Model(nn.Module):
        def __init__(
            self,
            param_1=10,
            param_2=20
        ): ...
    ```
    then the metadata file should read:
    ```
    parameters:
        param_1: 10
        param_2: 20
    ```

    - In `index_to_label.yaml` one should specify the decoding, for that particular model, from indices (i.e. the class the model will spit out) to actual labels (i.e. names). The file needs to be organised similarly to the following

    ```
    0: telephone
    1: person speaking
    ...
    ```