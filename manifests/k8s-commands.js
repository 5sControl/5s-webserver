const util = require('util');
const exec = util.promisify(require('child_process').exec);

const deployments = {
    front: './deployments/5scontrol-front-deployment.yaml',
    algorithmController: './deployments/algorithms-controller-deployment.yaml',
    idleModel: './deployments/idle-model-deployment.yaml',
    machineJsModel: './deployments/machine-js-model-deployment.yaml',
    machineModel: './deployments/machine-model-deployment.yaml',
    minMaxModel: './deployments/min-max-model-deployment.yaml',
    onvif: './deployments/onvif-deployment.yaml',
    onvifFinder: './deployments/onviffinder-deployment.yaml',
    backend: './deployments/z-django-deployment.yaml',
}

async function runCommandAsync(command) {
  try {
    const { stdout, stderr } = await exec(command);
    console.log('stdout:', stdout);
    console.error('stderr:', stderr);
    return stdout;
  } catch (error) {
    console.error('Error:', error);
  }
}

async function remove(path) {
  try {
    const res = await runCommandAsync(`sudo kubectl delete -f ${path}`);
    console.log(res, 'res')
    return res;
  } catch (error) {
    console.error('Error:', error);
  }
}

async function apply(path) {
  try {
    const res = await runCommandAsync(`sudo kubectl apply -f ${path}`);
    console.log(res, 'res')
    return res;
  } catch (error) {
    console.error('Error:', error);
  }
}

async function reload(service) {
  try {
    if (deployments[service]) {
       const res = await remove(deployments[service]);
        const res2 = await apply(deployments[service]);
        console.log(res, 'res')
        return true;
    } else {
        return null;
    }
  } catch (error) {
    console.error('Error:', error);
  }
}

module.exports = {reload}

