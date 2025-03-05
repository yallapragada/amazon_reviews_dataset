## How to build and push the image

docker build -t my_registry/no-fsdp-trainer:v1 .
docker push my_registry/no-fsdp-trainer:v1

## Apply the PyTorchJob in Kubeflow
kubectl apply -f pytorchjob.yaml

## Check the status
kubectl get pytorchjob

## View training jobs
kubectl logs -f job/no-fsdp-trainer -c pytorch