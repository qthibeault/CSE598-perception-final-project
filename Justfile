image-name := "qthibeault/ros"
image-tag := "humble"
mnt-path := "/workspace"

default:

build-image:
    docker build \
        --tag {{image-name}}:{{image-tag}} \
        --build-arg USER_UID=$(id -u $USER) \
        $PWD

build-modules:
    @echo "building modules"

run-module module:
    @echo "running module {{module}}"

shell:
    - docker run \
        --rm \
        --interactive \
        --tty \
        --mount "type=bind,src=$PWD,dst={{mnt-path}}" \
        --workdir {{mnt-path}} \
        --user $(id -u $USER) \
        {{image-name}}:{{image-tag}} \
        /bin/bash
