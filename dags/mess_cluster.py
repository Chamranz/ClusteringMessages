from docker.types import Mount
from airflow.providers.docker.operators.docker import DockerOperator


dockerops_kwargs = {
    "mount_tmp_dir": False,
    "mounts": [
        Mount(
            source='/Users/kamrankurbanov/PycharmProjects/ClusterPsycho/data',
            target='/opr/airflow/data/',
            type="bind",
        )
    ]
}

raw_data_path = "/opt/airflow/data/raw/data__{{ ds }}.csv"
raw_model_path = "/opt/airflow/data/raw/lda__{{ ds }}.pkl"
result_data_path = "/opt/airflow/data/clusterising/result__{{ ds }}.png"

from airflow.decorators import dag
from airflow.utils.dates import days_ago

@dag("new_messages", start_date = days_ago(0), schedule='@daily', catchup = False)
def taskflow():
    # Первая задача
    messages_load = DockerOperator(
        task_id = "messages_load",
        container_name = "task__messages__load",
        image = "messages-loader:latest",
        command = f"python Parsing.py --data_path {raw_data_path}, --page_number 6",
        **dockerops_kwargs
    )

    # Вторая задача
    lda_model = DockerOperator(
        task_id = "lda_model",
        container_name = "task__lda__model",
        image = "model-train:latest",
        command = f"python TrainLDA.py --n_topics 6 --data_path {raw_data_path} --model_path {raw_model_path}" ,
        **dockerops_kwargs
    )

    #Последняя задача

    cluster_make = DockerOperator(
        task_id = "cluster_make",
        container_name = "task__cluster__make",
        image = "cluster-kmeans:latest",
        command = f'python kmeans_clusters.py --P 3 --n_clusters 9 --model_path {raw_model_path} --data_path {raw_data_path} --save_pic {result_data_path}',
        **dockerops_kwargs
    )

    messages_load >> lda_model >> cluster_make

taskflow()
