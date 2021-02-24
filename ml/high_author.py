# !/usr/bin/python
# -*- coding: utf-8 -*-
def recieve_recall_result():
    logger.info('----------------[recieve_recall_result] start----------------')
    prv = request.json
    # prv = request.values
    logger.info(f'[recall data]::[{prv}]')
    status = prv.get("Status")
    code = prv.get("Code")
    jobid = prv.get("JobId")
    message = prv.get("Message")
    if status == "Succ":
        datastr = prv.get("Data")
        data = json.loads(datastr)
        nsq_data = {
            "action": "insert",
            "sync_biz": SYNC_BIZ,
            "sync_time": get_milli_timestamp(),
            "columns": [
                {
                    "id": str(data.get(VID)),
                    TO_NSQ_TAG: int(data.get(TAG, 0)),
                    TO_NSQ_SCORE: float(data.get(SCORE, 0.00))
                }
            ],
            "feature_columns": {
                str(data.get(VID)): [
                    {
                        "FeatureName": f"d_st_2_{TO_NSQ_TAG}",
                        "FeatureValue": {
                            "StringValue": str(data.get(TAG, 0))
                        }
                    },
                    {
                        "FeatureName": f"d_st_2_{TO_NSQ_SCORE}",
                        "FeatureValue": {
                            "FloatValue": data.get(SCORE, 0.00)
                        }
                    }
                ]
            }
        }
        push_nsq(data.get(VID), SYNC_BIZ, model_result=None, nsq_data=nsq_data)
        logger.info(f'[cont quality]::[jobid {jobid}] produce: {nsq_data}')
    return ''

def push_nsq(content_id, model_name, model_result=None, nsq_data=None):
    if not nsq_data:
        data = {
            "action": "insert",
            "sync_biz": SYNC_BIZ,
            "sync_time": get_timestamp(),
            "columns": [
                {
                    "id": str(content_id),
                    TAG: model_result.get(TAG, 0),
                    SCORE: model_result.get(SCORE, 0.0)
                }]
        }
    else:
        data = nsq_data
    # url = 'http://%s/pub?topic=%s' % ('172.25.20.245:4151', 'test_data_sync_feature')
    url = 'http://%s/pub?topic=%s' % ('172.16.42.95:4151', 'data_sync_feature')
    try:
        response = request(url, data)
        if response is None or len(response) == 0:
            logger.info("write %s data to nsq error, content_id:%s" % (model_name, str(content_id)))
            return 1
    except Exception as e:
        logger.info("write %s to nsq error, content_id:%s, err: %s" % (model_name, str(content_id), str(e)))
        send_msg_to_oa(NOTICE_NAME_LIST, f'[{get_localip_str()}]::[write {content_id} to nsq err.]')
        return 1
    # log.system.info('url: %s' % url)
    logger.info('publish %s data, topic: %s, action: insert,  data: %s, success!' % (model_name, p_topic, data))
    return 0

if __name__ == '__main__':

