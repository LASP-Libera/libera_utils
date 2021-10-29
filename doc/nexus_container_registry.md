# Nexus Docker Container Registry


## Basics


Nexus is an artifact storage service run by the LASP webteam. It contains an area that we use as a container
registry, similar to an AWS ECR or Dockerhub. You will need to obtain WebIAM permissions from the webteam to 
access Nexus and its web UI is only available from inside LASP's network (e.g. via VPN).

Nexus is located in the DMZ at https://artifacts.pdmz.lasp.colorado.edu/


## Usage


Fundamentally, all that the container registry does is provide a web API that the docker cli can interface with. 
Every docker command that interacts with a registry is just communicating with Nexus via HTTP. The documentation of
this API is here (knowledge of the API spec is not required for pushing docker images): 
https://docs.docker.com/registry/spec/api/

The URL for the Nexus container registry is:

_Internal:_ `docker-registry.pdmz.lasp.colorado.edu`

_External:_ `lasp-registry.colorado.edu`

**Note:** At time of writing, the external registry URL is having problems 
due to a load balancer issue on LASP's network. It's recommended that you use the internal URL, which requires you to be
on the LASP network.


### Docker Login


1. Check that you are not already logged in by running `cat ~/.docker/config.json` and ensuring that neither 
Nexus registry URL is in the list of logged in registries.

2. Run `docker login docker-registry.pdmz.lasp.colorado.edu` (you will be prompted for your WebIAM username and password).

3. You should see a success message and your `~/.docker/config.json` file should now contain a reference to the
    registry url.


### Pulling and Pushing


Images in a docker registry are uniquely identified by their URL path. "Repositories" within a registry are separated
only by path segments and permissions. That is, there is nothing special about the `libera` directory in the registry
vs `libera/mysubdir/myimage`. You can push images to any location for which you have permissions.

To push an image that was created locally as `my-image:my-tag`:
1. Re-tag the image with the registry URL by running:

    ```
    docker tag my-image:my-tag docker-registry.pdmz.lasp.colorado.edu/libera/my-image:my-tag
    ```

    The location to which it will be pushed is entirely defined by the tag you give it.

3. Assuming you are already logged in to the registry, run:

    ```
    docker push docker-registry.pdmz.lasp.colorado.edu/libera/my-image:my-tag
    ```

    **Note:** If there is already an image at this location in the registry with the same tag (but a different SHA),
    the tag will be removed from the old image and it will remain there but "untagged". You can verify the association
    between an image manifest (uniquely identified by its SHA) and a tag by inspecting the tag in the Nexus web UI.


To pull an image that is in the Nexus registry as `my-image:my-tag`, run
```shell
docker pull docker-registry.pdmz.lasp.colorado.edu/libera/my-image:my-tag
```


### Re-Tagging


For example, you just pushed `my-image:v1.1.1` but you also want it tagged as `latest` so it becomes the default image.
To tag an image with a second tag, simply tag it locally (`docker tag ...`) and push it again. Assuming the image 
has not changed, it will associate the manifest with the new tag (the old tag association will also remain).
