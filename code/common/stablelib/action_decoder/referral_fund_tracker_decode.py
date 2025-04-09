import pandas as pd
import numpy as np
import enum

class TransactionType(enum.Enum):
    approve = 0
    transfer = 1
    transferDai = 1.1
    stake = 2
    deposit = 2.1
    referral = 3
    withdraw = 4

class ActionType(enum.Enum):
    unknown = -1
    stake = 0
    referral = 1
    withdraw = 2


class ReferralTransferDecoded:

    def __init__(self,row:pd.Series, caller_address = None) -> None:
        jtype = None
        self.action:ActionType=ActionType.unknown
        event_name = row['event_name']
        transferJson = row['dl']
        self.code = None
        self.contract = row['contract_address']
        if event_name.lower().strip() == 'transfer':
            if 'from' in transferJson or '_from' in transferJson:
                jtype = TransactionType.transfer
            elif 'spender' in transferJson:
                jtype = TransactionType.approve
            elif 'dst' in transferJson:
                jtype = TransactionType.transferDai
        elif event_name.lower().strip() == 'staked':
            jtype = TransactionType.stake
            self.action=ActionType.stake
        elif event_name.lower().strip() == 'deposit':
            jtype = TransactionType.deposit
            self.action=ActionType.stake
        elif event_name.lower().strip() == 'referral':
            if 'assets' in transferJson:
                jtype = TransactionType.deposit
            else:
                jtype = TransactionType.stake
            self.action=ActionType.referral
        elif event_name.lower().strip() == 'withdrawn' or event_name.lower().strip() == 'withdraw':
            if 'assets' in transferJson:
                jtype = TransactionType.deposit
            else:
                jtype = TransactionType.stake
            self.action=ActionType.withdraw
        if(jtype is None):
            print("Unknown event",event_name=='Stakes', event_name, transferJson)
            return None
        self.type = jtype
        self.event_name = event_name
        self.caller_address = caller_address
        self.perspective = None
        self.tx_hash:str = row['tx_hash']
        self.row = row
        self.counter_value = None
        self.burn = None
        self.mint = None
        self.event_index =row['event_index']  if 'event_index' in row.keys() else 1e8
        self.dataFromJsonAndType(transferJson, jtype)
    
    def dataFromJsonAndType(self, transferJson, jtype):
        if jtype == TransactionType.transfer:
            self.from_address = transferJson.get('from', transferJson.get('_from',None))
            self.to_address = transferJson.get('to', transferJson.get('_to',None))
            self.value = transferJson.get('value', transferJson.get('amount', None)) 
        elif jtype == TransactionType.approve: # Approve is not tested
            self.from_address = transferJson.get('owner', None)
            self.to_address = transferJson.get('spender', None)
            self.value = transferJson.get('value', None)
        elif jtype == TransactionType.transferDai:
            self.from_address = transferJson.get('src', None)
            self.to_address = transferJson.get('dst', None)
            self.value = transferJson.get('wad', None)
        elif jtype == TransactionType.stake:
            if self.action == ActionType.referral:
                self.from_address = transferJson.get('user', None)
                self.to_address = self.contract
                self.value = transferJson.get('amount', None)
                self.code = transferJson.get('referral', 'untagged')
            elif self.action == ActionType.withdraw:
                self.to_address = transferJson.get('user', None)
                self.from_address = self.contract
                self.value = transferJson.get('amount', None)
            else:
                self.from_address = transferJson.get('user', None)
                self.to_address = self.contract
                self.value = transferJson.get('amount', None) 
                self.action = ActionType.stake
                self.code = 'untagged'
        elif jtype == TransactionType.deposit:
            if self.action == ActionType.withdraw:
                self.to_address = transferJson.get('owner', None)
                self.from_address = self.contract
                self.value = transferJson.get('shares', None)
                self.counter_value = transferJson.get('assets', None)
            else:
                self.from_address = transferJson.get('owner', None)
                self.to_address = self.contract
                self.value = transferJson.get('shares', None)
                self.counter_value = transferJson.get('assets', None)
                self.code = transferJson.get('referral', 'untagged')
            # if transferJson.get('receiver'):
            #     if transferJson.get('owner')!=transferJson.get('receiver'):
            #         print("Different owner and receiver",self.tx_hash, transferJson)
        else:
            print("Unknown type", jtype, transferJson, self.event_name)
            return np.nan
        
        if self.from_address is None or self.to_address is None or self.value is None:
            print("ERROR PARSING from/to/value",self.from_address, self.to_address, self.value)
            print("ERROR PARSING",self.tx_hash, transferJson, self.contract, self.row)
            return np.nan
        self.from_address = self.from_address.lower()
        self.to_address = self.to_address.lower()
        self.burn = self.to_address == '0x0000000000000000000000000000000000000000'
        self.mint = self.from_address == '0x0000000000000000000000000000000000000000'
        if self.value is not None:
            self.value = int(self.value)
        else:
            print("ERROR", transferJson)
            self.value = np.nan
        return self

    def set_perspective(self, tracking_address):
        if self.from_address == tracking_address:
            self.perspective = 'out'
            # self.value = -self.value
        elif self.to_address == tracking_address:
            self.perspective = 'in'
        else:
            print("Unknown perspective", tracking_address, self)
            self.perspective = 'unknown'
        return self
    
    def __str__(self) -> str:
        return f"({self.event_index}){self.from_address} -> {self.to_address} : {self.value} ({self.type}/{self.action}/{self.perspective})"