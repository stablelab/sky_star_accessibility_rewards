# %%
import uuid 


# %%
# %%
# !pip install itables -y

import datetime
reward_percentage = 0.4/100
checkFrom = datetime.datetime.fromisoformat('2030-12-20 23:59:59')

import numpy as np

# %%
import json
import time
import pandas as pd

import os
from os.path import join, dirname
from dotenv import load_dotenv
base = os.path.abspath("")
import sys
sys.path.append(os.path.join(base))
sys.path.append(os.path.join(base,'./'))
# import common.stablelib.decoder.input_decoder as decoder


dotenv_path = join(dirname(os.path.dirname('__file__')), '.env')
load_dotenv(dotenv_path)

# %%
import snowflake.connector
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
# build connecting string from .env
from common.config import (
    DATABASE_NAME,
    DATABASE_HOST,
    DATABASE_USER,
    DATABASE_PASSWORD,
    DATABASE_PORT,
    SNOWFLAKE_ACCOUNT,
    SNOWFLAKE_USER,
    SNOWFLAKE_PASSWORD,
    SNOWFLAKE_DATABASE
)

import common.slogger as slogger
import common.stablelib.action_decoder.referral_fund_tracker_decode as rd
import common.stablelib.slack_interaction.sendAlert as si
import common.stablelib.stableapi_integration.resetCache as stapi_reset

query_sUSDSprices="""SELECT DISTINCT ON (DATE_TRUNC('month', date)) token_id, date, price
FROM coingecko.market_data
where token_id = 'susds'
ORDER BY DATE_TRUNC('month', date) DESC, date DESC
"""

stlogger = slogger.logger
stlogger.info('loaded simple_server.py')
stlogger.info("loading dotenv %s from %s", dotenv_path, base)

potnum=0
def main(runId,logger):
    stlogger.info("Snowflake user: %s", SNOWFLAKE_USER)
    def slog(*args):
        stlogger.info(f'RUNID {runId}|'+'|'.join([str(a) for a in args]))

    global potnum
    potnum=0
    snowflake_conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        database=SNOWFLAKE_DATABASE,
    )

    connection = "postgresql://"+DATABASE_USER+":"+DATABASE_PASSWORD+"@"+DATABASE_HOST+":"+DATABASE_PORT+"/"+DATABASE_NAME

    engine = create_engine(connection) 

     ## get sUSDS prices
    dfsUSDSPrices = pd.read_sql(query_sUSDSprices, engine)
    dfsUSDSPrices['month'] = dfsUSDSPrices['date'].dt.to_period('M')
    slog('GOT SUSDS prices')

    # %% [markdown]
    # # track funds

    # %%

    all_timer = time.time()

    
    logger.sendAlertWithMrkdwnBlocks(["Starting REFTRACK"])
    slog("!!TRACKING FUNDS!!")
    slog("DATE:", datetime.datetime.now())
    slog("!!GETTING DATA!!")
    timer = time.time()
    susds_logs = """
        SELECT * FROM ethereum.core.ez_decoded_event_logs WHERE contract_address = lower('0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD') AND tx_status = 'SUCCESS'
    """
    dfSUsdsLogs = pd.read_sql(susds_logs, snowflake_conn)
    dfSUsdsLogs.columns = dfSUsdsLogs.columns.str.lower()
    dfSUsdsLogs['dl'] = dfSUsdsLogs['decoded_log'].apply(lambda x:json.loads(x))
    dfSUsdsLogs=dfSUsdsLogs[[ 'block_timestamp','block_number', 'tx_hash',
            'event_name', 'dl','contract_address', 'contract_name','event_index', 'origin_function_signature', 'origin_from_address',
        'origin_to_address', 'topics']]
    slog("!!USDS DATA LOADED!!", time.time()-timer, dfSUsdsLogs.shape)
    logger.sendAlertWithMrkdwnBlocks(["Data Loaded", f"Time taken: _{(time.time()-timer):5.2f}_ s", f"Rows: _{dfSUsdsLogs.shape[0]}_"])

    # %%
    timer = time.time()

    cowswap_logs_query = """
        SELECT *
        FROM ethereum.core.ez_decoded_event_logs AS del
        WHERE EXISTS (
            SELECT 1
            FROM ethereum.core.ez_decoded_event_logs AS del1
            WHERE del1.tx_hash = del.tx_hash
            AND del1.contract_address = LOWER('0x9008d19f58aabd9ed0d60971565aa8510560ab41')
        )
        AND EXISTS (
            SELECT 1
            FROM ethereum.core.ez_decoded_event_logs AS del2
            WHERE del2.tx_hash = del.tx_hash
            AND del2.contract_address IN (
                LOWER('0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD'),
                LOWER('0x0650CAF159C5A49f711e8169D4336ECB9b950275'),
                LOWER('0x10ab606B067C9C461d8893c47C7512472E19e2Ce')
            )
            AND del2.tx_status = 'SUCCESS'
        )
    """
    cowswap_logs = pd.read_sql(cowswap_logs_query, snowflake_conn)
    cowswap_logs.columns = cowswap_logs.columns.str.lower()
    deposit_staked_df = cowswap_logs[(cowswap_logs['event_name'] == 'Deposit') | (cowswap_logs['event_name'] == 'Staked')]
    deposit_staked_df = deposit_staked_df[deposit_staked_df['contract_address'].isin([
        '0xa3931d71877c0e7a3148cb7eb4463524fec27fbd',
        '0x0650caf159c5a49f711e8169d4336ecb9b950275',
        '0x10ab606b067c9c461d8893c47c7512472e19e2ce'
    ])]


    synthetic_referals = deposit_staked_df.copy()[0:0]
    for index, row in deposit_staked_df.iterrows():
        if row['event_name'] == 'Deposit':
            referral = row.copy()
            tjson = json.loads(row['decoded_log'])
            tjson['referral'] = 1003
            referral['event_name'] = 'Referral'
            # cowswap is 1004
            referral['decoded_log'] = json.dumps(tjson)
            synthetic_referals = synthetic_referals._append(referral)
        if row['event_name'] == 'Staked':
            staked = row.copy()
            amount = json.loads(row['decoded_log'])['amount']
            user = json.loads(row['decoded_log'])['user']
            staked['decoded_log'] = '{"amount": "' + amount + '", "referral": 1003, "user": "' + user + '"}'
            synthetic_referals = synthetic_referals._append(staked)
    synthetic_referals['dl'] = synthetic_referals['decoded_log'].apply(lambda x:json.loads(x))
    synthetic_referals=synthetic_referals[[ 'block_timestamp','block_number', 'tx_hash',
            'event_name', 'dl','contract_address', 'contract_name','event_index', 'origin_function_signature', 'origin_from_address',
        'origin_to_address', 'topics']]

    synthetic_referals['event_name']= synthetic_referals['event_name'].str.lower()
    slog("!!COWSWAP DATA LOADED!!", time.time()-timer, synthetic_referals.shape)
    logger.sendAlertWithMrkdwnBlocks(["Synthetic Data Loaded", f"Time taken: _{(time.time()-timer):5.2f}_ s", f"Rows: _{synthetic_referals.shape[0]}_"])

    timer = time.time()
    lazysummer_logs_query = """
        SELECT * FROM ethereum.core.ez_decoded_event_logs
        WHERE tx_hash IN (
            SELECT d.tx_hash
            FROM ethereum.core.ez_decoded_event_logs d
            WHERE d.contract_address IN ('0xa3931d71877c0e7a3148cb7eb4463524fec27fbd', '0x0650caf159c5a49f711e8169d4336ecb9b950275', '0x10ab606b067c9c461d8893c47c7512472e19e2ce')
            AND d.event_name = 'Deposit'
            AND EXISTS (
                -- Check if the same transaction has a "Rebalanced" event (from any contract)
                SELECT 1
                FROM ethereum.core.ez_decoded_event_logs r
                WHERE r.tx_hash = d.tx_hash
                AND r.event_name = 'Rebalanced'
            )
        )
    """
    lazysummer_logs_query = pd.read_sql(lazysummer_logs_query, snowflake_conn)
    print("lazysummer_logs_query done")
    # column names to lower
    lazysummer_logs_query.columns = lazysummer_logs_query.columns.str.lower()
    # filter by event_name = 'Deposit' or event_name = 'Staked'
    deposit_staked_df = lazysummer_logs_query[(lazysummer_logs_query['event_name'] == 'Deposit') | (lazysummer_logs_query['event_name'] == 'Staked')]
    deposit_staked_df = deposit_staked_df[deposit_staked_df['contract_address'].isin([
        '0xa3931d71877c0e7a3148cb7eb4463524fec27fbd',
        '0x0650caf159c5a49f711e8169d4336ecb9b950275',
        '0x10ab606b067c9c461d8893c47c7512472e19e2ce'
    ])]
    synthetic_referals_lazysummer = deposit_staked_df.copy()[0:0]
    # delete all rows, keep head
    for index, row in deposit_staked_df.iterrows():
        if row['event_name'] == 'Deposit':
            referral = row.copy()
            amount = json.loads(row['decoded_log'])['shares']
            user = json.loads(row['decoded_log'])['owner']
            referral['event_name'] = 'Referral'
            # cowswap is 1004
            referral['decoded_log'] = '{"amount": "' + amount + '", "referral": 1016, "user": "' + user + '"}'
            synthetic_referals_lazysummer = synthetic_referals_lazysummer._append(referral)
        if row['event_name'] == 'Staked':
            staked = row.copy()
            amount = json.loads(row['decoded_log'])['amount']
            user = json.loads(row['decoded_log'])['user']
            referral['decoded_log'] = '{"amount": "' + amount + '", "referral": 1016, "user": "' + user + '"}'
            synthetic_referals_lazysummer = synthetic_referals_lazysummer._append(staked)
    synthetic_referals_lazysummer

    synthetic_referals_lazysummer['dl'] = synthetic_referals_lazysummer['decoded_log'].apply(lambda x:json.loads(x))
    synthetic_referals_lazysummer=synthetic_referals_lazysummer[[ 'block_number','block_timestamp', 'tx_hash',
            'event_name', 'dl','contract_address', 'contract_name','event_index', 'origin_function_signature', 'origin_from_address',
        'origin_to_address', 'topics']]

    synthetic_referals_lazysummer['event_name']= synthetic_referals_lazysummer['event_name'].str.lower()
    synthetic_referals = pd.concat([synthetic_referals, synthetic_referals_lazysummer])

    logger.sendAlertWithMrkdwnBlocks(["Synthetic Lazysummer Data Loaded", f"Time taken: _{(time.time()-timer):5.2f}_ s", 
                                      f"Rows: _{synthetic_referals.shape[0]}_",
                                        synthetic_referals['dl'].apply(lambda x: x['referral'] if isinstance(x,dict) and 'referral' in x else 'broken').value_counts().to_markdown()])
    # %%

    timer = time.time()
    contracts = ['0x0650caf159c5a49f711e8169d4336ecb9b950275', '0x10ab606B067C9C461d8893c47C7512472E19e2Ce','0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD']
    slog("analysing CONTRACTS", contracts)
    #make lowercase
    contracts = [x.lower() for x in contracts]
    sky_farm_logs_all = """
        SELECT * FROM ethereum.core.ez_decoded_event_logs WHERE contract_address IN (%s)
    """ % ",".join(["('"+x.lower()+"')" for x in contracts])
    dfSFLogs = pd.read_sql(sky_farm_logs_all, snowflake_conn)
    dfSFLogs.columns = dfSFLogs.columns.str.lower()
    dfSFLogs['dl'] = dfSFLogs['decoded_log'].apply(lambda x:json.loads(x))
    dfSFLogs=dfSFLogs[[ 'block_timestamp', 'block_number', 'tx_hash',
            'event_name', 'dl','contract_address', 'contract_name','event_index', 'origin_function_signature', 'origin_from_address',
        'origin_to_address', 'topics'
        ]]
    dfSFLogs['event_name']= dfSFLogs['event_name'].str.lower()
    #get deposits
    dfDeposits = dfSFLogs[(dfSFLogs['event_name'] == 'referral') & (dfSFLogs['contract_address'] == contracts[-1])]
    dfDeposits
    slog("!!LOG FARM DATA LOADED!!", time.time()-timer)
    logger.sendAlertWithMrkdwnBlocks(["Farm Event Data Loaded", f"Time taken: _{(time.time()-timer):5.2f}_ s", f"Farm event Rows: _{dfSFLogs.shape[0]}_"])

    # %%
    timer = time.time()

    # dfSFLogs.loc[dfSFLogs['event_name'] == 'deposit', 'event_name'] = 'staked'

    # %%
    dfSFLogs['event_name'] = dfSFLogs['event_name'].str.lower()
    dfSUsdsLogs['event_name'] = dfSUsdsLogs['event_name'].str.lower()

    na_count = dfSFLogs['block_timestamp'].isna().sum()
    dfSFLogs = dfSFLogs.sort_values(by=['block_number','event_index'], ascending=False)
    dfSFLogs['block_timestamp'] = dfSFLogs['block_timestamp'].interpolate()
    logger.sendAlertWithMrkdwnBlocks(["Filled NAs was %s -> %s" % (na_count, dfSFLogs['block_timestamp'].isna().sum())])
    # %%
    dfSUsdsLogsT = dfSUsdsLogs[dfSUsdsLogs['event_name'] == 'transfer']

    # %%
    dfUSDSB =  dfSFLogs[dfSFLogs['event_name'].isin(['withdrawn','deposit','withdraw']) & (dfSFLogs['contract_address'].str.lower() == contracts[-1].lower())]
    # generate sanity-check event relative to stake or withdraw
    dfUSDSB['am'] = dfUSDSB.apply(lambda x: int(x['dl']['shares'])if x['event_name'] == 'deposit' else -1*int(x['dl']['shares']), axis=1)
    # dfUSDSB

    # %%
    dfSFStakeWithdraw = dfSFLogs[dfSFLogs['event_name'].isin(['withdrawn','staked']) & (dfSFLogs['contract_address'].str.lower() == contracts[0].lower())]

    dfSFStakeWithdraw['am'] = dfSFStakeWithdraw.apply(lambda x: int(x['dl']['amount'])if x['event_name'] == 'staked' else -1*int(x['dl']['amount']), axis=1)

    dfRelEventsRef = dfSFLogs[dfSFLogs['event_name'].isin(['referral','staked','transfer','deposit','withdrawn','withdraw'])]
    # dfSFStakeWithdraw.sort_values('block_timestamp', inplace=True, ascending=True)
    dfTraceEvents = pd.concat([dfRelEventsRef,synthetic_referals])
    dfTraceEvents.sort_values(['block_timestamp','event_index'], inplace=True, ascending=True)
    dfTraceEvents.reset_index(inplace=True)
    # fvi = dfTraceEvents[dfTraceEvents['event_name'] !='transfer'].first_valid_index()
    # fvi
    # dfTraceEvents = dfTraceEvents.iloc[fvi:]
    dfTraceEvents['event_name'].value_counts()

    logger.sendAlertWithMrkdwnBlocks(["Data Loaded", f"Time taken: _{(time.time()-timer):5.2f}_ s", f"Rows: _{dfTraceEvents.shape[0]}_", f"Events: _{dfTraceEvents['event_name'].value_counts()}"])
    # %%
    potnum=0
    def create_new_pot(referral_code,address,amount, tx_hash, contract,liquidity_token,previous_tx_trace=[],previous_pot_trace = [], spawn = None):
        global potnum
        pot = {
            'potnum': potnum,
            'referral_code': referral_code,
            'address': address,
            'amount': amount,
            'tx_hash': tx_hash,
            'spawn': spawn,
            'initial_amount': amount,
            'contract': contract,
            'liquidity_token': liquidity_token,
            'TXtrace': previous_tx_trace + [tx_hash],
            'POTtrace': previous_pot_trace
        }
        potnum += 1
        return pot
    def find_pot(from_address, ps, code=None, liquidity_token=None, contract=None):
        for p in ps:
            if (
                p['address'] == from_address 
                and (code == None or p['referral_code'] == code) 
                and p['amount'] > 0 
                and (liquidity_token == p['liquidity_token'] or liquidity_token == None)
                and ( contract == None or contract == p['contract'] )
            ):
                return p
        return None
            


    # %%
    def contract_to_token(contract):
        if contract == '0x0650caf159c5a49f711e8169d4336ecb9b950275':
            return '0x0650caf159c5a49f711e8169d4336ecb9b950275'
        elif contract == '0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD':
            return 'sUSD'
        elif contract == '':
            slog("NO CONTRACT")
            return 'sUSD'
        else:
            return contract
    # %%
    idxcnt = 0
    pots = []
    gunstaked = {c:0 for c in contracts}
    gwdraw = {c:0 for c in contracts}
    gstaked = {c:0 for c in contracts}

    # %%
    potsOverTime = {}

    # %%
    log_list = []
    def add_log(tx,timestamp,pot_id, integrator,pool,from_address,amount_delta, action):
        log = {
            'tx':tx,
            'timestamp':timestamp,
            'pot_id':pot_id,
            'integrator':integrator,
            'pool':pool,
            'from_address':from_address,
            'amount_delta':amount_delta,
            'action':action
        }
        log_list.append(log)
    import copy

    # %%
    def checkPots(pots,cutoff):

        dfPots = pd.DataFrame(pots)
        # dfPots['amount'].sum() / 1e18
        
        dfRelEventsRefc = dfRelEventsRef[dfRelEventsRef['block_timestamp'] <= cutoff]
        error = False
        global contracts
        for c in range(len(contracts)):
            error = checkContract(c,dfPots,dfRelEventsRefc) or error
        return error

    # %%
    def withdrawFromPot(parsedEvent:rd.ReferralTransferDecoded, currentTxHash,pots,p_contract=None, 
                        p_liquidity_token=None,change_type='undefined'):
        balance_changes = []
        transfer_changes = []
        pot_og_address = parsedEvent.to_address if change_type == 'withdraw' else parsedEvent.from_address
        end = False
        f_pot = find_pot(pot_og_address, pots, contract=p_contract, liquidity_token=p_liquidity_token)
        transferred=0
        booking_num = 0
        gwdraw[parsedEvent.contract] += parsedEvent.value
        refcode=None
        if f_pot is not None:
            contract = f_pot['contract']
            liquidity_token = f_pot['liquidity_token']
            left = parsedEvent.value
            refcode = f_pot['referral_code']
            reduce = min(left, f_pot['amount'])
            left -= reduce
            transferred += reduce
            trace = {'pot':f_pot['potnum'], 'tx':currentTxHash,  'amount':reduce, 'type':change_type}
            tx_tace = [f_pot['TXtrace']]
            f_pot['amount'] -= reduce
            add_log(currentTxHash, parsedEvent.row['block_timestamp'],f_pot['potnum'], f_pot['referral_code'], 
                    contract_to_token(parsedEvent.contract), parsedEvent.from_address, -1*reduce, change_type)
            # f_pot['POTtrace'].append(trace)
            while left>0 and f_pot:
                f_pot = find_pot(pot_og_address, pots,contract=p_contract, liquidity_token=p_liquidity_token)
                if f_pot is not None:
                    contract = f_pot['contract']
                    if f_pot['referral_code'] != refcode:
                        if change_type == 'transfer':
                            slog('New code: Different referral code in transfer', currentTxHash)
                            p =create_new_pot(refcode,parsedEvent.to_address,transferred, 
                                                        parsedEvent.tx_hash, contract,liquidity_token, spawn='transfer', 
                                                        previous_pot_trace=trace)
                            pots.append(p)
                            add_log(currentTxHash, parsedEvent.row['block_timestamp'], p['potnum'], refcode, 
                                contract_to_token(parsedEvent.contract), parsedEvent.from_address, parsedEvent.value, 'transfer')
                            transfer_change = {'ref_code':str(refcode), 
                                            'amount':transferred, 
                                            'timestamp':parsedEvent.row['block_timestamp'],
                                            'block_number':parsedEvent.row['block_number'],
                                            'event_index':parsedEvent.event_index,
                                            'from':parsedEvent.from_address,
                                            'to':parsedEvent.to_address,
                                            'type':'transfer',
                                            'booking_num':booking_num,
                                            'contract':parsedEvent.contract, 'tx':currentTxHash}
                            transfer_changes.append(transfer_change)
                            booking_num += 1
                    # slog('New code: Different referral code in withdraw', currentTxHash)
                    # pots.append(create_new_pot(refcode,pot_og_address,transferred, parsedEvent.tx_hash, parsedEvent.contract,liquidity_token, spawn='transfer'))
                            transferred = 0
                            trace = []
                            tx_tace = []
                            refcode = f_pot['referral_code']
                        else:
                            balance_change = {'ref_code':str(refcode), 'amount':transferred * -1, 
                                            'timestamp':parsedEvent.row['block_timestamp'],
                                            'block_number':parsedEvent.row['block_number'],
                                            'event_index':parsedEvent.event_index,
                                            'from':parsedEvent.from_address,
                                            'type':change_type,
                                            'booking_num':booking_num,
                                            'contract':parsedEvent.contract, 'tx':currentTxHash}
                            balance_changes.append(balance_change)
                            booking_num += 1
                            transferred = 0
                            refcode = f_pot['referral_code']
                    reduce = min(left, f_pot['amount'])
                    left -= reduce
                    add_log(currentTxHash, parsedEvent.row['block_timestamp'], f_pot['potnum'], f_pot['referral_code'], 
                            contract_to_token(parsedEvent.contract), parsedEvent.from_address, -1*reduce, change_type)
                    transferred += reduce
                    f_pot['amount'] -= reduce
                    trace = {'pot':f_pot['potnum'],'tx':currentTxHash,  'amount':reduce, 'type':change_type}
                    # f_pot['POTtrace'].append(trace)
                    tx_tace.append(f_pot['TXtrace'])
                else:
                    slog('ERROR: Not enough funds in pot withdraw', currentTxHash,'left',left/1e18,'withdraw from', pot_og_address)
                    if left/1e18 > 1:
                        end=True
                    else:
                        slog('ERROR too small, ignored')
                    break
            if change_type=='transfer':
                if left>0:
                    slog("ERROR left over in transfer", currentTxHash, left)
                # last_f_trace.append(trace)
                p=create_new_pot(refcode,parsedEvent.to_address,transferred, parsedEvent.tx_hash, 
                                            parsedEvent.contract,liquidity_token, spawn='transfer',
                                            previous_pot_trace=trace)
                pots.append(p)
                add_log(currentTxHash, parsedEvent.row['block_timestamp'], p['potnum'], refcode, 
                            contract_to_token(parsedEvent.contract), parsedEvent.from_address, parsedEvent.value, 'transfer')
                transfer_change = {'ref_code':str(refcode), 
                                            'amount':transferred,
                                            'type':'transfer', 
                                            'timestamp':parsedEvent.row['block_timestamp'],
                                            'block_number':parsedEvent.row['block_number'],
                                            'event_index':parsedEvent.event_index,
                                            'from':parsedEvent.from_address,
                                            'to':parsedEvent.to_address,
                                            'booking_num':booking_num,
                                            'contract':parsedEvent.contract, 'tx':currentTxHash}
                transfer_changes.append(transfer_change)
                booking_num += 1
            else:
                balance_change = {'ref_code':str(refcode), 
                                'amount':transferred * -1, 
                                'timestamp':parsedEvent.row['block_timestamp'],
                                'contract':parsedEvent.contract, 
                                'block_number':parsedEvent.row['block_number'],
                                'event_index':parsedEvent.event_index,
                                'type': change_type,
                                'from':parsedEvent.from_address,
                                'booking_num':booking_num,
                                'tx':currentTxHash}
                balance_changes.append(balance_change)
                booking_num += 1
                transferred = 0
                if left>0:
                    slog("ERROR left over in withdraw", currentTxHash, left)
                # balance_change = {'ref_code':refcode, 'amount':transferred * -1, 'timestamp':parsedEvent.row['block_timestamp'],
                #                   'contract':parsedEvent.contract, 'tx':currentTxHash}
                # balance_changes.append(balance_change)
        else:
                slog('ERROR: No pot found in ',change_type, currentTxHash, parsedEvent)
                if parsedEvent.value/1e18 > 1:
                    end=True
                else:
                    slog('ERROR too small, ignored', parsedEvent.value/1e18)
        
        return end, balance_changes, transfer_changes


    ##generate empty pot and start and end of month checkpoints
    all_refs = dfTraceEvents['dl'].apply(lambda x: str(x.get('referral', 'untagged'))).dropna().astype(str).unique().tolist()
    all_refs.sort()

    slog("USING REFS:",all_refs, "of", dfTraceEvents.shape)
    # code_contract_uniques = dfBH[['contract','ref_code']].value_counts().reset_index()
    import itertools
    def empty_for_all(timestamp):
        ret = []
        for perm in itertools.product(contracts,all_refs):
            ret.append( {'ref_code': str(perm[1]),
            'amount': 0,
            'timestamp': timestamp,
            'contract': perm[0],
            'type':'checkpoint',
            'tx': f'virtual_{timestamp}_{perm[0]}_{perm[1]}'})
        return ret
    # %%


    # %%
    timer = time.time()
    end = False
    currentMonth = dfTraceEvents.iloc[0]['block_timestamp'].to_period('M')
    lastTime = dfTraceEvents.iloc[0]['block_timestamp']
    lastHash = None
    balance_history = []
    transfer_history = []
    # balance_history.extend(empty_for_all(lastTime))
    balance_history.extend(empty_for_all(currentMonth.to_timestamp()))
    while dfTraceEvents.shape[0] > 0 and not end:
        nextMonth = dfTraceEvents.iloc[0]['block_timestamp'].to_period('M')
        if nextMonth != currentMonth:
            balance_history.extend(empty_for_all(nextMonth.to_timestamp()-pd.Timedelta('1s')))
            potsOverTime[currentMonth.strftime('%Y-%m')] = [copy.deepcopy(p) for p in pots]
            balance_history.extend(empty_for_all(nextMonth.to_timestamp()))
            slog('MONTH SAVED', currentMonth, '->', nextMonth, 'Augmented Timelines', nextMonth.to_timestamp()-pd.Timedelta('1s'),nextMonth.to_timestamp())
            currentMonth = nextMonth
        currentTxHash = dfTraceEvents.iloc[0]['tx_hash']
        currentTxEvents = dfTraceEvents[dfTraceEvents['tx_hash'] == currentTxHash]
        if currentTxEvents['block_timestamp'].max() > lastTime and checkFrom < lastTime:
            check = checkPots(pots,lastTime)
            if check:
                slog('ERROR: Pots do not match events', 'last tx', lastHash, 'current tx', currentTxHash)
                break
        to_process_events:list[rd.ReferralTransferDecoded] = []
        idxcnt +=currentTxEvents.shape[0]
        # if 'withdraw' in currentTxEvents['event_name'].values and contracts[-1] in currentTxEvents['contract_address'].values:
        #    end = True
        if idxcnt % 1000 == 0:
            slog(idxcnt)
        
        if 'staked' in currentTxEvents['event_name'].values or 'deposit' in currentTxEvents['event_name'].values:
            eventCandidate = currentTxEvents[currentTxEvents['event_name'].str.lower().isin(['staked','deposit'])]
            if eventCandidate.shape[0] == 1:
                to_process_events.append(rd.ReferralTransferDecoded(eventCandidate.iloc[0]))
            elif eventCandidate.shape[0] > 1:
                to_process_events = to_process_events + [rd.ReferralTransferDecoded(r[1]) for r in eventCandidate.iterrows()]
                # slog('Multiple staked events in current tx', currentTxHash, to_process_events)
            else:
                slog('No staked events in current tx', currentTxHash)
            # if 'deposit' in currentTxEvents['event_name'].values:
            #    end=True
        if 'transfer' in currentTxEvents['event_name'].values:
            eventCandidate = currentTxEvents[currentTxEvents['event_name'].str.lower().isin(['transfer'])]
            # if eventCandidate.shape[0] == 1:
            #    to_process_events.append(eventCandidate.iloc[0])
            # elif eventCandidate.shape[0] > 1:
            refs = [rd.ReferralTransferDecoded(r[1]) for r in eventCandidate.iterrows()]
            to_process_events = to_process_events + [r for r in refs if not r.mint and not r.burn]
                # slog('Multiple transfer events in current tx', currentTxHash)
        if 'referral' in currentTxEvents['event_name'].values:
            eventCandidate = currentTxEvents[currentTxEvents['event_name'].str.lower().isin([ 'referral'])]
            refs = []
            if eventCandidate.shape[0] == 1:
                refs.append(rd.ReferralTransferDecoded(eventCandidate.iloc[0]))
            elif eventCandidate.shape[0] > 1:
                for r in eventCandidate.iterrows():
                    refs.append(rd.ReferralTransferDecoded(r[1]))
                # slog('Multiple events in current tx', currentTxHash)
            else:
                slog('No Farm events in current tx', currentTxHash)
            for r in refs:
                # for e in to_process_events:
                #    #remove accroding stake event
                ## possibility of 2 idnetical events cauding issues here exists, it has however not been observed to date.
                ## the problem why it cannot be done more elegant is that the synthetic event dont have a linear number and therefore no order.
                # By improving htat methodology this could also be improved.
                to_process_events = [x for x in to_process_events if not( 
                x.action == rd.ActionType.stake and x.from_address == r.from_address and x.value == r.value)]
                to_process_events.append(r)
                # end=True
                
        if 'withdrawn' in currentTxEvents['event_name'].values or  'withdraw' in currentTxEvents['event_name'].values:
            eventCandidate = currentTxEvents[currentTxEvents['event_name'].str.lower().isin(['withdrawn','withdraw'])]
            # if eventCandidate.shape[0] == 1:
            #    to_process_events.append(eventCandidate.iloc[0])
            # elif eventCandidate.shape[0] > 1:
            for r in eventCandidate.iterrows():
                to_process_events.append(rd.ReferralTransferDecoded(r[1]))
                # slog('Multiple events in current tx', currentTxHash)
            # else:
            #    slog('No Farm events in current tx', currentTxHash)

        
                
        # if to_process_events == []:
        #    slog('No events in current tx', currentTxHash)
            # end = True
        to_process_events = sorted(to_process_events, key=lambda x: x.event_index)
        for e in to_process_events:
            parsedEvent = e
            if parsedEvent.code is not None and str(parsedEvent.code) not in all_refs:
                slog('!!ERROR!!: New UNKNOWN referral code in tx', currentTxHash,parsedEvent.code)
            if parsedEvent.action == rd.ActionType.referral:
                if parsedEvent.code is not None:
                    # slog('REFERRAL: tx', currentTxHash, parsedEvent.code)
                    gstaked[parsedEvent.contract] += parsedEvent.value
                    p=create_new_pot(
                    parsedEvent.code,
                    parsedEvent.from_address,
                    parsedEvent.value, 
                    parsedEvent.tx_hash, 
                    parsedEvent.contract,
                    contract_to_token(parsedEvent.contract),
                    previous_pot_trace=[{'pot':'spawn', 'tx':currentTxHash, 'amount':parsedEvent.value}],
                        spawn = 'ref')
                    balance_change = {'ref_code':parsedEvent.code, 
                                    'amount':parsedEvent.value, 
                                    'timestamp':parsedEvent.row['block_timestamp'],
                                    'block_number':parsedEvent.row['block_number'],
                                    'event_index':parsedEvent.event_index,
                                    'to':parsedEvent.from_address,
                                    'type':'referral',
                                    'contract':parsedEvent.contract, 
                                    'booking_num':0,
                                    'tx':currentTxHash}
                    balance_history.append(balance_change)
                    pots.append(p)
                    add_log(currentTxHash, parsedEvent.row['block_timestamp'], p['potnum'], parsedEvent.code, contract_to_token(parsedEvent.contract), parsedEvent.from_address, parsedEvent.value, 'referral')
                else:
                    slog('ERROR: No referral code in tx', currentTxHash)
            elif parsedEvent.action == rd.ActionType.withdraw:
                    # slog('WITHDRAW: tx', currentTxHash)
                    f_pot = find_pot(parsedEvent.to_address, pots, contract=parsedEvent.contract)
                    tend, balance_changes, transfer_changes = withdrawFromPot(parsedEvent, currentTxHash, pots,p_contract=parsedEvent.contract, change_type='withdraw')
                    balance_history.extend(balance_changes)
                    transfer_history.extend(transfer_changes)
                    end = end or tend
            elif parsedEvent.action == rd.ActionType.stake:
                # slog('NEW UNNAMED POT')
                gstaked[parsedEvent.contract] += parsedEvent.value
                p = create_new_pot(
                    parsedEvent.code,
                    parsedEvent.from_address,
                    parsedEvent.value, 
                    parsedEvent.tx_hash, 
                    parsedEvent.contract, 
                    contract_to_token(parsedEvent.contract),
                    previous_pot_trace=[{'pot':'spawn', 'tx':currentTxHash, 'amount':parsedEvent.value}],
                    spawn='stake')
                balance_change = {'ref_code':parsedEvent.code, 
                                'amount':parsedEvent.value, 
                                'timestamp':parsedEvent.row['block_timestamp'],
                                'block_number':parsedEvent.row['block_number'],
                                'event_index':parsedEvent.event_index,
                                'from':parsedEvent.from_address,
                                'type':'stake',
                                'booking_num':0,
                                'contract':parsedEvent.contract, 'tx':currentTxHash}
                balance_history.append(balance_change)
                pots.append(p)
                add_log(currentTxHash, parsedEvent.row['block_timestamp'], p['potnum'], parsedEvent.code, 
                        contract_to_token(parsedEvent.contract), parsedEvent.from_address, parsedEvent.value, 'stake')
            # elif parsedEvent.type == rd.TransactionType.deposit:
            #    if parsedEvent.action == rd.ActionType.referral:
            #       if  parsedEvent.action == rd.ActionType.stake:
            #          gstaked += parsedEvent.value
            #          pots.append(create_new_pot(
            #             parsedEvent.code,
            #             parsedEvent.from_address,
            #             parsedEvent.value,
            #             parsedEvent.tx_hash, 
            #             parsedEvent.contract, 
            #             contract_to_token(parsedEvent.contract),
            #             spawn='stake'))

            elif parsedEvent.type == rd.TransactionType.transfer:
                liquidity_token = parsedEvent.contract
                tend, _, transfer_changes = withdrawFromPot(parsedEvent, currentTxHash,pots, p_liquidity_token=liquidity_token, change_type='transfer')
                end = end or tend
                transfer_history.extend(transfer_changes)
        dfTraceEvents = dfTraceEvents[dfTraceEvents['tx_hash']!=(currentTxHash)]
        lastTime = currentTxEvents['block_timestamp'].max()
        lastHash = currentTxEvents['tx_hash'].max()


    if end:
        slog('ERROR: Pots do not match events', 'last tx', lastHash, 'current tx', currentTxHash)
        logger.sendAlertWithMrkdwnBlocks(["ERROR: Pots do not match events, STOPPED", 'last tx', lastHash, 'current tx', currentTxHash])
        exit(1)
    potsOverTime[currentMonth.strftime('%Y-%m')] = [copy.deepcopy(p) for p in pots]
    slog('FUNDS TRACKED', time.time()-timer)
    logger.sendAlertWithMrkdwnBlocks(["Funds Tracked", f"Time taken: _{(time.time()-timer):5.2f}_ s"])

    # %%
    dfBH = pd.DataFrame(balance_history)
    dfBH['ref_code'] = dfBH['ref_code'].astype(str)
    code_contract_uniques = dfBH[['contract','ref_code']].value_counts().reset_index()

    total_seconds = 60 * 60 * 24 * 365
    slog("using reward %:",reward_percentage, "on total seconds:", total_seconds)

   
    def rowToUSDValue(row):
        #safety check to prevent accidental wrong usage,
        # hence DF requires the contract field as well
        if row['contract'] == '0xa3931d71877c0e7a3148cb7eb4463524fec27fbd'.lower():
            month_price = dfsUSDSPrices[dfsUSDSPrices['month'] == row['month']]['price']
            return row['eligible_tvl'] * month_price.values[0] if month_price.shape[0]>0 else row['eligible_tvl']
        else:
            print("wrong address called")
            return row['eligible_tvl']
    # %%
    dfBHs = []
    dfBHsR = []
    dfRewards = None
    i = 0
    for r in code_contract_uniques.iterrows():
        current_contract = r[1]['contract']
        dfBHt = dfBH[(dfBH['contract'] == current_contract) & (dfBH['ref_code'] == r[1]['ref_code'])].copy()
        dfBHt.sort_values('timestamp',inplace=True)
        dfBHsR.append(dfBHt)
        dfBHt.loc[:,'cumsum'] = dfBHt['amount'].cumsum()
        dfBHt.loc[:,'held_time'] = dfBHt['timestamp'].diff().replace(pd.NaT,pd.Timedelta(seconds=1)).dt.total_seconds()/total_seconds
        dfBHt.loc[:,'eligible_tvl'] = dfBHt['cumsum'] * dfBHt['held_time']
        dfBHt.loc[:,'month'] = dfBHt['timestamp'].dt.to_period('M')
        if current_contract == '0xa3931d71877c0e7a3148cb7eb4463524fec27fbd':
            testTail = dfBHt['eligible_tvl'].tail().copy()
            slog(dfBHt['eligible_tvl'].tail())
            dfBHt.loc[:,'eligible_tvl'] = dfBHt.apply(lambda x: rowToUSDValue(x), axis=1)
            slog('sUSDS adapted', current_contract, r[1]['ref_code'], i)
            slog(dfBHt['eligible_tvl'].tail())
            try:
                slog('diff', dfBHt['eligible_tvl'].tail()/testTail)
            except Exception as e:
                slog("FAILED", e)
                slog(testTail)
        dfReward = (dfBHt.groupby('month')['eligible_tvl'].sum()/1e18).reset_index()
        # dfReward.reset_index(inplace=True)
        dfReward.loc[:,'contract'] = current_contract
        dfReward.loc[:,'ref_code'] = r[1]['ref_code']
        if dfRewards is None:
            dfRewards = dfReward
        else:
            dfRewards = pd.concat([dfRewards,dfReward])
        dfBHs.append(dfBHt)
        slog('___________', current_contract, r[1]['ref_code'], i,'___________')
        i+=1
        
    dfRewards.rename(columns={'ref_code':'referral_code'},inplace=True)
    dfRewards['eligible_tvl'] = dfRewards['eligible_tvl'].astype(np.float64)
    dfRAgg = dfRewards.groupby(['month', 'referral_code'])['eligible_tvl'].sum().reset_index()

    slog("Done calculating rewards", dfRAgg.index)
    # %%
    import common.stablelib.sqlutils.sqlutils as sqlutils

    ctable = sqlutils.CustomTable('acc_rew_eligible_payout_amounts_continous_agg',dfRAgg,schema_name='sky', primary_keys=['month','referral_code'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfRAgg)

    dfRAggPayout = dfRAgg
    dfRAggPayout['payout'] = dfRAggPayout['eligible_tvl'] * reward_percentage
    dfRAggPayout

    import common.stablelib.sqlutils.sqlutils as sqlutils

    ctable = sqlutils.CustomTable('acc_rew_eligible_payout_amounts_continous_agg_payout',dfRAggPayout,schema_name='sky', primary_keys=['month','referral_code'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfRAggPayout)


    import common.stablelib.sqlutils.sqlutils as sqlutils

    ctable = sqlutils.CustomTable('acc_rew_eligible_payout_amounts_continous',dfRewards,schema_name='sky', primary_keys=['month','referral_code', 'contract'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfRewards)

    slog("Done inserting rewards")
    # %%
    dfLogList = pd.DataFrame(log_list)

    # %% [markdown]
    # # safety Check

    # %%
    ps = pd.DataFrame(pots)

    # %%
    ps['liquidity_token'].value_counts()

    # %%

    dfPots = pd.DataFrame(pots)
    dfPots['amount'].sum() / 1e18

    # %%
    def rowToAmount(row):
        if row['event_name'] == 'deposit':
            return int(row['dl']['shares'])
        elif row['event_name'] == 'withdraw':
            return -1*int(row['dl']['shares'])
        elif row['event_name'] == 'withdrawn':
            return -1*int(row['dl']['amount'])
        elif row['event_name'] == 'staked':
            return int(row['dl']['amount'])
        else:
            return 0

    # %%

    def checkContract(conNum):
        dfRelEventsRef[(dfRelEventsRef['contract_address'] ==contracts[conNum])]['event_name'].value_counts()
        stakeSum = (dfRelEventsRef[(dfRelEventsRef['event_name'] != 'referral') & (dfRelEventsRef['contract_address'] ==contracts[conNum])  ].apply(lambda x: rowToAmount(x),axis=1).sum() / 1e18)
        stakeSum
        computed_sum = dfPots[dfPots['contract'] == contracts[conNum]]['amount'].sum() / 1e18
        if abs( stakeSum - (computed_sum))>1:
            return " ".join([str(x) for x in ['\n🔴!!ERROR!!🔴\n Contract', contracts[conNum], "stake sum mismatch",
                                            stakeSum, (computed_sum),
                                            'OFF BY:', stakeSum - computed_sum,
                                            "=>", f'{ 100*(abs(stakeSum / computed_sum)-1):3.5f}%']])
        else:
            return " ".join([str(x) for x in ['✅OK!!✅ Contract', contracts[conNum], "stake sum MATCH",
                                            stakeSum, computed_sum]])


    # %%
    slog('\n!!SANITY CHECKS!!')
    logs = []
    for c in range(len(contracts)):
        logs.append(checkContract(c))
    for l in logs:
        slog(l)
    logger.sendAlertWithMrkdwnBlocks(["Sanity Checks", *logs])
    slog('!!SANITY CHECKS!!\n')


    # %%
    dfTH = pd.DataFrame(transfer_history)

    # %%
    dfBTH = pd.DataFrame(balance_history + transfer_history)

    # %%
    dfBTHRel = dfBTH[dfBTH['type']!= 'checkpoint']

    # %%
    acc_ev_tab = sqlutils.CustomTable('acc_rew_logs_events_typed_num',dfBTHRel,schema_name='sky', primary_keys=['tx','block_number','event_index','booking_num'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    acc_ev_tab.create_table()
    acc_ev_tab.insert(dfBTHRel, 100000)

    # %% [markdown]
    # # Plotting

    # %%
    import common.stablelib.sqlutils.sqlutils as sqlutils

    slog('!!COMMITING TO DB!!')
    timer = time.time()

    allDF = None
    for m in potsOverTime:
        dfPots = pd.DataFrame(potsOverTime[m])
        dfPots['amount'].sum() / 1e18
        perIntegrator = dfPots.groupby('referral_code')['amount'].sum()/1e18
        perIntegrator = perIntegrator.reset_index()

        perIntegrator['amount'] = perIntegrator['amount'].astype(int)
        perIntegrator['referral_code'] = perIntegrator['referral_code'].astype(str)
        # perIntegrator = perIntegrator[perIntegrator['referral_code'] != 'unknown']
        # chart.create_bar_graph(perIntegrator, 'referral_code', 'amount','Split stake by Referral Code %s'%m, 'Code', 'Total Staked Amount',plot_numbers='int', number_units='USDS')
        perIntegrator['month'] = m
        if allDF is None:
            allDF = perIntegrator
        else:
            allDF = pd.concat([allDF, perIntegrator])
    pAllDF = allDF.pivot(index='referral_code', columns='month', values='amount').fillna(0)
    pAllDF
    pAllDfDiff = pAllDF.diff(axis=1).fillna(0)
    slog("all diffs",pAllDfDiff)

    c0 = pAllDfDiff.columns[0]
    slog("setting c0",c0)
    pAllDfDiff[c0] = pAllDF[c0]
    #Check if we need losspots
    msg = "Loss pots needed" if np.any(pAllDfDiff <0) else "No loss pots needed"
    slog(msg)
    def rowToAmount(row):
        if row['event_name'] == 'referral':
            if 'shares' in row['dl']:
                return int(row['dl']['shares'])/1e18
            else:
                return int(row['dl']['amount'])/1e18
        return 0
    dfAllSFLogs = pd.concat([dfSFLogs, synthetic_referals])
    dfGross = dfAllSFLogs[(dfAllSFLogs['event_name'] == 'referral')]
    dfGross['gross'] = dfGross.apply(lambda x: rowToAmount(x),axis=1)
    dfGross['month'] = dfGross['block_timestamp'].dt.to_period('M').astype(str)
    dfGross['referral_code'] = dfGross['dl'].apply(lambda x: x['referral'])
    dfGrossPivot = dfGross.groupby(['month','referral_code'])['gross'].sum().reset_index()
    dfGrossPivot
    # smf = synthetic_referals
    # smf['m']= smf['block_timestamp'].dt.to_period('M').astype(str)
    # smf['s'] = smf['dl'].apply(lambda x: int(x['shares']))
    # smf[['m','s']].groupby('m').sum()


    opdir = os.path.join(base,'output')
    opdir

    dfNetPlot = pAllDF.reset_index().melt(id_vars=["referral_code"], var_name="month", value_name="dnet")
    dfNetPlot['month'] = dfNetPlot['month'].astype(str)
    dfNetPlot.sort_values('month', inplace=True)
    # dfNetPlot= dfNetPlot[dfNetPlot['referral_code'] != 'unknown']
    dfP = dfNetPlot[dfNetPlot['referral_code'] != 'untagged']

    ##redo per pool
    dfGrossPivot = dfGross.groupby(['month','referral_code','contract_address'])['gross'].sum().reset_index()
    dfGrossPivot.rename(columns={'contract_address':'pool'}, inplace=True)
    ctable = sqlutils.CustomTable('acc_rew_gross_amounts',dfGrossPivot,schema_name='sky', primary_keys=['month','referral_code','pool'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfGrossPivot)

    dfDeltaPlot = pAllDfDiff.reset_index().melt(id_vars=["referral_code"], var_name="month", value_name="dnet")
    dfDeltaPlot['month'] = dfDeltaPlot['month'].astype(str)
    dfDeltaPlot.sort_values('month', inplace=True)
    dfDeltaPlot= dfDeltaPlot[dfDeltaPlot['referral_code'] != 'unknown']
    dfP = dfDeltaPlot[dfDeltaPlot['referral_code'] != 'untagged']

    ##implement loss pots
    dfDeltaPayout = dfDeltaPlot.copy()[0:0]
    for rc in dfDeltaPlot['referral_code'].unique():
        dfP = dfDeltaPlot[dfDeltaPlot['referral_code'] == rc]
        dfP.reset_index(inplace=True)
        dfP.sort_values('month', inplace=True)
        currentRow = dfP.iloc[0]
        # dfP = dfP[dfP['month'] != currentRow['month']]
        current_losspot = 0
        for i,d in dfP.iterrows():
            if i == 0:
                continue
            nextRow = d
            slog(i, d['month'], current_losspot)
            if nextRow['dnet'] < 0:
                current_losspot += nextRow['dnet']
                dfP.at[i,'dnet'] = 0
                slog('LOSS POT NEEDED', rc, nextRow['month'], current_losspot)
            else:
                if nextRow['dnet']>current_losspot:
                    slog(i, dfP.shape)
                    dfP.at[i,'dnet'] += current_losspot
                    current_losspot = 0
                else:
                    current_losspot += nextRow['dnet']
                    dfP.at[i,'dnet'] = 0
                slog('LOSS POT APPLIED', rc, nextRow['month'], current_losspot)
        
        dfDeltaPayout = pd.concat([dfDeltaPayout, dfP], ignore_index=True) 
            

    dfDeltaPayout.rename(columns={'dnet':'amount'}, inplace=True)
    dfDeltaPayout.drop(columns=['index'], inplace=True)
    ctable = sqlutils.CustomTable('acc_rew_eligible_payout_amounts',dfDeltaPayout,schema_name='sky', primary_keys=['month','referral_code'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfDeltaPayout)

    # %% [markdown]
    # ## Add per farm stats

    # %%
    slog('!!PLOTTING NET!!')
    allDF = None
    for m in potsOverTime:
        dfPots = pd.DataFrame(potsOverTime[m])
        perIntegrator = dfPots.groupby(['referral_code','liquidity_token'])['amount'].sum()/1e18
        perIntegrator = perIntegrator.reset_index()

        perIntegrator['amount'] = perIntegrator['amount'].astype(int)
        perIntegrator['referral_code'] = perIntegrator['referral_code'].astype(str)
        # perIntegrator = perIntegrator[perIntegrator['referral_code'] != 'unknown']
        # chart.create_bar_graph(perIntegrator, 'referral_code', 'amount','Split stake by Referral Code %s'%m, 'Code', 'Total Staked Amount',plot_numbers='int', number_units='USDS')
        perIntegrator['month'] = m
        if allDF is None:
            allDF = perIntegrator
        else:
            allDF = pd.concat([allDF, perIntegrator])
    pAllDF = None
    dfDeltaPlot = None
    for l in allDF['liquidity_token'].unique():
        dfL = allDF[allDF['liquidity_token'] == l]
        dfL = dfL.pivot(index='referral_code', columns='month', values='amount').fillna(0)
        dfLD = dfL.diff(axis=1).fillna(0)
        dfLD[c0] = dfL[c0]
        dfLDd = dfLD.reset_index().melt(id_vars=["referral_code"], var_name="month", value_name="dnet")
        dfLDd['pool'] = l
        dfLDd['month'] = dfLDd['month'].astype(str)
        if pAllDF is None:
            pAllDF = dfL
        else:
            pAllDF = pd.concat([pAllDF, dfL], axis=1)
        if dfDeltaPlot is None:
            dfDeltaPlot = dfLDd
        else:
            dfDeltaPlot = pd.concat([dfDeltaPlot, dfLDd])
    dfDeltaPlot.sort_values('month', inplace=True)
    ctable = sqlutils.CustomTable('acc_rew_net_delta_amounts',dfDeltaPlot,schema_name='sky', primary_keys=['month','referral_code','pool'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfDeltaPlot)
    dfDeltaPlot['pool'] = dfDeltaPlot['pool'].replace({
        '0x0650CAF159C5A49f711e8169D4336ECB9b950275'.lower():'Sky Farm',
        '0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD'.lower(): 'sUSDS Farm',
        '0x10ab606B067C9C461d8893c47C7512472E19e2Ce'.lower(): 'Chronicle'
    })
    dfNetPlot = allDF
    dfNetPlot['pool']=dfNetPlot['liquidity_token']
    logger.sendAlertWithMrkdwnBlocks(["Net Stake", dfNetPlot.iloc[-10:,:].to_markdown()])
    ctable = sqlutils.CustomTable('acc_rew_net_amounts',dfNetPlot,schema_name='sky', primary_keys=['month','referral_code','pool'],default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine)
    ctable.create_table()
    ctable.insert(dfNetPlot)
    dfNetPlot['pool'] = dfNetPlot['pool'].replace({
        '0x0650CAF159C5A49f711e8169D4336ECB9b950275'.lower():'Sky Farm',
        '0xa3931d71877C0E7a3148CB7Eb4463524FEc27fbD'.lower(): 'sUSDS Farm',
        '0x10ab606B067C9C461d8893c47C7512472E19e2Ce'.lower(): 'Chronicle'
    })

    logger.sendAlertWithMrkdwnBlocks(["Insert Data Done Stake", f"Time taken: _{(time.time()-timer):5.2f}_ s"])

    # %%
    stapi_reset.resetCache(True,True)
    logger.sendAlertWithMrkdwnBlocks(["Reset Cache, *DONE*",f"time taken: _{(time.time()-all_timer):5.2f}_ s"])
    sqlutils.CustomTable('dashboard_updates',
                     columns={'last_update':'DATETIME', 'dashboard_id':'TEXT', 'active':'BOOLEAN'},schema_name='metadata', primary_keys=['dashboard_id'],
                     default_strategy=sqlutils.Strategy.OVERWRITE, engine=engine, snake_case=False).insert([(datetime.datetime.now(),'ar_integrators_eth',True), (datetime.datetime.now(),'ar_integrators_payout_eth',True)])
    slog("RUN FINISHED", runId, "Time taken:", time.time()-all_timer)

    snowflake_conn.close()
    engine.dispose()

def call_script():
    runId = 'REFTRACK '+ str( uuid.uuid4())
    logger = si.SlackLogger('alert', runId)
    complete = False
    for i in range(2):
        if complete:
            break
        logger.sendAlert("Starting Referral Tracker %s, trial: %s"%(runId, i))
        try:
            main(runId, logger)
            complete = True
        except Exception as e:
            logger.sendAlertWithMrkdwnBlocks(["ERROR", f"Error: {str(e)}"])
            print(e)
            raise e