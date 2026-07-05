"""Dataset-specific counterfactual-agent prompt content.

Each dataset has its own closed-set root-cause vocabulary and its own
telemetry schema (file layout, CSV columns, timestamp units). The two are
genuinely different per dataset, so they are kept as separate data here
rather than collapsed into one shared string. This module only consolidates
the file layout (previously basic_prompt_Bank.py / basic_prompt_Telecom.py
were separate modules with the same shape).
"""

from typing import Dict, Tuple

_BANK_CAND = """## POSSIBLE ROOT CAUSE REASONS:

- high CPU usage
- high memory usage
- network latency
- network packet loss
- high disk I/O read usage
- high disk space usage
- high JVM CPU load
- JVM Out of Memory (OOM) Heap

## POSSIBLE ROOT CAUSE COMPONENTS:

- apache01
- apache02
- Tomcat01
- Tomcat02
- Tomcat04
- Tomcat03
- MG01
- MG02
- IG01
- IG02
- Mysql01
- Mysql02
- Redis01
- Redis02"""

_BANK_SCHEMA = f"""## TELEMETRY DIRECTORY STRUCTURE:

- You can access the telemetry directory in our microservices system: `dataset/Bank/telemetry/`.

- This directory contains subdirectories organized by a date (e.g., `dataset/Bank/telemetry/2021_03_05/`).

- Within each date-specific directory, you’ll find these subdirectories: `metric`, `trace`, and `log` (e.g., `dataset/Bank/telemetry/2021_03_05/metric/`).

- The telemetry data in those subdirectories is stored in CSV format (e.g., `dataset/Bank/telemetry/2021_03_05/metric/metric_container.csv`).

## DATA SCHEMA

1.  **Metric Files**:

    1. `metric_app.csv`:

        ```csv
        timestamp,rr,sr,cnt,mrt,tc
        1614787440,100.0,100.0,22,53.27,ServiceTest1
        ```

    2. `metric_container.csv`:

        ```csv
        timestamp,cmdb_id,kpi_name,value
        1614787200,Tomcat04,OSLinux-CPU_CPU_CPUCpuUtil,26.2957
        ```

2.  **Trace Files**:

    1. `trace_span.csv`:

        ```csv
        timestamp,cmdb_id,parent_id,span_id,trace_id,duration
        1614787199628,dockerA2,369-bcou-dle-way1-c514cf30-43410@0824-2f0e47a816-17492,21030300016145905763,gw0120210304000517192504,19
        ```

3.  **Log Files**:

    1. `log_service.csv`:

        ```csv
        log_id,timestamp,cmdb_id,log_name,value
        8c7f5908ed126abdd0de6dbdd739715c,1614787201,Tomcat01,gc,"3748789.580: [GC (CMS Initial Mark) [1 CMS-initial-mark: 2462269K(3145728K)] 3160896K(4089472K), 0.1985754 secs] [Times: user=0.59 sys=0.00, real=0.20 secs] "
        ```

{_BANK_CAND}

## CLARIFICATION OF TELEMETRY DATA:

1. This microservice system is a banking platform.

2. The `metric_app.csv` file only contains four KPIs: rr, sr, cnt, and mrt,. In contrast, `metric_container.csv` records a variety of KPIs, such as CPU usage and memory usage. The specific names of these KPIs can be found in the `kpi_name` field.

3. In different telemetry files, the timestamp units and cmdb_id formats may vary:

- Metric: Timestamp units are in seconds (e.g., 1614787440).

- Trace: Timestamp units are in milliseconds (e.g., 1614787199628).

- Log: Timestamp units are in seconds (e.g., 1614787201).

4. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""

_TELECOM_CAND = """## POSSIBLE ROOT CAUSE REASONS:

- CPU fault
- network delay
- network loss
- db connection limit
- db close

## POSSIBLE ROOT CAUSE COMPONENTS:

(if the root cause is at the node level, i.e., the root cause is a specific node)

- os_001
- os_002
- os_003
- os_004
- os_005
- os_006
- os_007
- os_008
- os_009
- os_010
- os_011
- os_012
- os_013
- os_014
- os_015
- os_016
- os_017
- os_018
- os_019
- os_020
- os_021
- os_022

(if the root cause is at the pod level, i.e., the root cause is a specific container)

- docker_001
- docker_002
- docker_003
- docker_004
- docker_005
- docker_006
- docker_007
- docker_008

(if the root cause is at the service level, i.e., if all pods of a specific service are faulty, the root cause is the service itself)

- db_001
- db_002
- db_003
- db_004
- db_005
- db_006
- db_007
- db_008
- db_009
- db_010
- db_011
- db_012
- db_013"""

_TELECOM_SCHEMA = f"""## TELEMETRY DIRECTORY STRUCTURE:

- You can access the telemetry directory in our microservices system: `dataset/Telecom/telemetry/`

- This directory contains subdirectories organized by a date (e.g., `dataset/Telecom/telemetry/2020_04_11/`).

- Within each date-specific directory, you’ll find these subdirectories: `metric` and `trace` (e.g., `dataset/Telecom/telemetry/2020_04_11/metric/`).

- The telemetry data in those subdirectories is stored in CSV format (e.g., `dataset/Telecom/telemetry/2020_04_11/metric/metric_container.csv`).

## DATA SCHEMA

1.  **Metric Files**:

    1. `metric_app.csv`:

        ```csv
        serviceName,startTime,avg_time,num,succee_num,succee_rate
        osb_001,1586534400000,0.333,1,1,1.0
        ```

    2. `metric_container.csv`:

        ```csv
        itemid,name,bomc_id,timestamp,value,cmdb_id
        999999996381330,container_mem_used,ZJ-004-060,1586534423000,59.000000,docker_008
        ```

    3. `metric_middleware.csv`:

        ```csv
        itemid,name,bomc_id,timestamp,value,cmdb_id
        999999996508323,connected_clients,ZJ-005-024,1586534672000,25,redis_003
        ```

    4. `metric_node.csv`:

        ```csv
        itemid,name,bomc_id,timestamp,value,cmdb_id
        999999996487783,CPU_iowait_time,ZJ-001-010,1586534683000,0.022954,os_017
        ```

    5. `metric_service.csv`:

        ```csv
        itemid,name,bomc_id,timestamp,value,cmdb_id
        999999998650974,MEM_Total,ZJ-002-055,1586534694000,381.902264,db_003
        ```

2.  **Trace Files**:

    1. `trace_span.csv`:

        ```csv
        callType,startTime,elapsedTime,success,traceId,id,pid,cmdb_id,dsName,serviceName
        JDBC,1586534400335,2.0,True,01df517164d1c0365586,407d617164d1c14f2613,6e02217164d1c14b2607,docker_006,db_003,
        LOCAL,1586534400331,6.0,True,01df517164d1c0365586,6e02217164d1c14b2607,8432217164d1c1442597,docker_006,db_003,local_method_017
        RemoteProcess,1586534400324,55.0,True,01df517164d1c0365586,8432217164d1c1442597,b755e17164d1c13f5066,docker_006,,csf_005
        FlyRemote,1586534400149,7.0,TRUE,fa1e817164d1c0375444,da74117164d1c0955052,b959f17164d1c08c5050,docker_003,,fly_remote_001
        OSB,1586534660846,376.0,True,d9c4817164d5baee6924,77d1117164d5baee6925,None,os_021,,osb_001
        ```

{_TELECOM_CAND}

## CLARIFICATION OF TELEMETRY DATA:

1. This service system is a telecom database system.

2. The `metric_app.csv` file only contains five KPIs: startTime, avg_time, num, succee_num, succee_rate. In contrast, other metric files record a variety of KPIs, such as CPU usage and memory usage. The specific names of these KPIs can be found in the `name` field.

3. In all telemetry files, the timestamp units and cmdb_id formats remain consistent:

- Metric: Timestamp units are in milliseconds (e.g., 1586534423000).

- Trace: Timestamp units are in milliseconds (e.g., 1586534400335).

4. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""

_PROMPTS: Dict[str, Tuple[str, str]] = {
    "bank": (_BANK_CAND, _BANK_SCHEMA),
    "telecom": (_TELECOM_CAND, _TELECOM_SCHEMA),
}


def get_basic_prompt(dataset: str) -> Tuple[str, str]:
    """Return (cand, schema) prompt text for a dataset.

    Bank gets its own vocabulary/schema; everything else (Telecom, Market)
    falls back to the Telecom prompt, matching prior behavior.
    """
    key = "bank" if str(dataset or "").strip().lower() == "bank" else "telecom"
    return _PROMPTS[key]
