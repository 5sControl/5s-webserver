const fastify = require('fastify')();
const k8s = require('@kubernetes/client-node');
const kc = new k8s.KubeConfig();
kc.loadFromDefault();

const {reload} = require('./k8s-commands');

const k8sApi = kc.makeApiClient(k8s.CoreV1Api);


const randomInt = () => {
  const min = 1;
  const max = 111111111111;
  return Math.floor(Math.random() * (max - min + 1)) + min;
};
const createPodManifest = (image, name, env) => {
  env.push({name: 'link_reports',value: 'http://django-service:8000/api/reports/report-with-photos/'});
  const modifiedName = name.replace(/_/g, '-');
  return {
  apiVersion: 'v1',
  kind: 'Pod',
  metadata: {
    name: randomInt().toString(),
  },
  spec: {
    containers: [
      {
        name: modifiedName,
        image: image,
        volumeMounts: [
            {
              name: 'images',
              mountPath: '/var/www/5scontrol/images',
            }
          ],
        env
      },
    ],
    volumes: [
        {
          name: 'images',
          persistentVolumeClaim: {
            claimName: 'images',
          },
        },
      ],
  },
};
}


const modifyPodAnnotation = (yamlFilePath) => {
  try {
    const fileContents = fs.readFileSync(yamlFilePath, 'utf8');

    const yamlObject = yaml.load(fileContents);
    if (yamlObject?.spec?.template?.metadata?.annotations?.["custom-label"]) {
      yamlObject.spec.template.metadata.annotations["custom-label"] = randomInt().toString();
    }

    const updatedYaml = yaml.dump(yamlObject);
    fs.writeFileSync(yamlFilePath, updatedYaml, 'utf8');
  } catch (e) {
    console.error('Error:', e);
  }
}


fastify.post('/create-pod', async (request, reply) => {
  try {
    console.log('start pod', request.body)
    const manifest = createPodManifest(request.body.image, request.body.name, request.body.envVars);
    const response = await k8sApi.createNamespacedPod('default', manifest);
    return {name: response.body.metadata.name};
  } catch (err) {
    console.error('Ошибка при создании Pod:', err);
    return `Ошибка при создании Pod: ${err.message}`;
  }
});

fastify.get('/deploy-pod', async (request, reply) => {
  try {
    if (!request.query?.service) {
      return 'service query param not found'
    }
    const response = await reload(request.query.service);
    if (!response) return `Pod not reloaded`
    return 'Pod reloaded';
  } catch (err) {
    console.log(err, 'err reload pod')
    return `reload pod error`;
  }
});

fastify.post('/stop-pod', async (request, reply) => {
  try {
    console.log('stop pod', request.body)
    const deleteResponse = await k8sApi.deleteNamespacedPod(request.body.pod, 'default', {});
    return {success: true}
  } catch (err) {
    console.error('Ошибка при остановке Pod:', err);
    return `Ошибка при остановке Pod: ${err.message}`;
  }
});

fastify.listen(4545, '0.0.0.0', (err, address) => {
  if (err) {
    console.error('Ошибка при запуске сервера:', err);
    process.exit(1);
  }
  console.log(`Сервер запущен на ${address}`);
});
