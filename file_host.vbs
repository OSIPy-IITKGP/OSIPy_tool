Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd.exe /k cd C:\Users\AN-74\Desktop\osi && python -m http.server 8010", 0, False