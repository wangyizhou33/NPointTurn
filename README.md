# Build this repo

1. build the dockerfile
    ```
    $ cd ${repo_root}  
    $ docker build -t n-point-turn .
    ```

2. set up  
    ```
    $ source setup.sh
    ```

3. build  
    ```
    $ cd build  
    $ build_main
      or
    $ build_test
    ```
