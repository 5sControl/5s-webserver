const fastify = require('fastify')();

const k8s = require('@kubernetes/client-node');
const kc = new k8s.KubeConfig();
kc.loadFromDefault();

const k8sApi = kc.makeApiClient(k8s.CoreV1Api);


const randomInt = () => {
  const min = 1;
  const max = 111111111111;
  return Math.floor(Math.random() * (max - min + 1)) + min;
};
const createPodManifest = (image, name, env) => {
  env.push({name: 'link_reports',value: 'http://192.168.1.136:80/api/reports/report-with-photos/'});
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
        image: '5scontrol/min_max_python:latest' || image,
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

fastify.post('/create-pod', async (request, reply) => {
  try {
    console.log('start pod', request.body)
    const manifest = createPodManifest(request.body.image, request.body.name, request.body.envVariables);
    const response = await k8sApi.createNamespacedPod('default', manifest);
    return {name: response.body.metadata.name};
  } catch (err) {
    console.error('Ошибка при создании Pod:', err);
    return `Ошибка при создании Pod: ${err.message}`;
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