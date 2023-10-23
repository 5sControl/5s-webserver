cd deployments && sudo kubectl delete -f . &&
sudo kubectl delete pods --all -n default &&
cd ../storage && sudo kubectl delete -f .
