import time
import sagemaker
from sagemaker.tensorflow import TensorFlow

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket_name = 'ditto-2500'

instance_type = 'ml.p3.8xlarge'
instance_count = 20
processes_per_host = 4

hyperparameters = {'epochs': 8,
                   'batch_size': 10000,
                   'learning-rate': 0.0001,
                   }

output_path = 's3://{}/output/'.format(bucket_name)

model_dir = output_path
job_name = 'sm-dist-{}x{}-workers'.format(instance_count, processes_per_host) + time.strftime('%Y-%m-%d-%H-%M-%S-%j', time.gmtime())

distributions = {'mpi': {
    'enabled': True,
    'processes_per_host': processes_per_host,
    'custom_mpi_options': '-verbose --NCCL_DEBUG=INFO -x OMPI_MCA_btl_vader_single_copy_mechanism=none -x NCCL_P2P_DISABLE=1'}
}

estimator_hvd = TensorFlow(
    base_job_name='ditto',
    entry_point='train_horovod_load.py',
    role=role,
    framework_version='2.1',
    py_version='py3',
    hyperparameters=hyperparameters,
    train_instance_count=instance_count,
    train_instance_type=instance_type,
    output_path=output_path,
    model_dir=model_dir,
    checkpoint_s3_uri=f'{output_path}/{job_name}/checkpoints',
    distributions=distributions,
)

train_path = 's3://{}/data/'.format(bucket_name)
estimator_hvd.fit({'train': train_path}, wait=True)
