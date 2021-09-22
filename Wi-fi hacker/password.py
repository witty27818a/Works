'''
What we are doing is
the same as the following command in cmd

netsh wlan show profile <network_name> key=clear
for every wifi.
'''

import subprocess
#so we can use system commands

import re
#so we can make use of regular expression

command_output = subprocess.run(["netsh", "wlan", "show", "profiles"], capture_output = True).stdout.decode('big5')

profile_names = (re.findall("所有使用者設定檔 : (.*)\r", command_output))

wifi_list = list()

if len(profile_names) != 0:
    for name in profile_names:
        wifi_profile = dict()
        profile_info = subprocess.run(["netsh", "wlan", "show", "profile", name], capture_output = True).stdout.decode('big5')
        if re.search("安全性金鑰             : 缺少", profile_info):
            continue
        else:
            wifi_profile["SSID名稱"] = name
            profile_info_pass = subprocess.run(["netsh", "wlan", "show", "profile", name, "key=clear"], capture_output = True).stdout.decode('big5')
            password = re.search("金鑰內容               : (.*)\r", profile_info_pass)
            if password == None:
                wifi_profile["金鑰內容"] = None
            else:
                wifi_profile["金鑰內容"] = password[1]
            wifi_list.append(wifi_profile)

for x in range(len(wifi_list)):
    print(wifi_list[x])

import time
time.sleep(5)