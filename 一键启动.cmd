@echo off
@chcp 936 >nul
rem 上面两行关闭命令回显并使用Windows简体中文代码页。
setlocal
rem 将本脚本设置限制在当前启动进程内。
cd /d "%~dp0"
rem 从解压目录启动，支持中文目录名和空格。
if not exist "%~dp0runtime\python.exe" goto missing_runtime
rem 运行前确认免安装解释器存在，避免误调用系统程序。
if not exist "%~dp0server.py" goto missing_source
rem 运行前确认程序主体存在，提示完整解压。
set "LEARNING_DESK_DATA="
rem 使用本包自己的课堂配置，避免继承开发环境的数据路径。
echo 正在启动神经网络学习台，请保留此窗口。
rem 告知启动状态，实际地址由服务选择空闲端口后显示。
"%~dp0runtime\python.exe" -E -s -B -X utf8 "%~dp0server.py" %*
rem 使用包内解释器和依赖，忽略外部Python配置，允许验收时传入不打开浏览器的参数。
if errorlevel 1 goto start_failed
rem 启动失败时保留错误窗口，正常退出则直接结束。
exit /b 0
rem 不改变调用方的命令窗口退出状态。
:missing_runtime
rem 统一处理运行环境缺失的情况。
echo 没有找到运行环境。请先完整解压整个文件夹，再双击“一键启动.cmd”。
rem 说明不能在压缩包内单独运行启动器。
pause
rem 等待用户查看缺失原因。
exit /b 1
rem 以失败状态返回，便于启动验收发现问题。
:missing_source
rem 统一处理程序文件缺失的情况。
echo 没有找到程序文件。请重新完整解压安装包。
rem 提供与实际检查结果对应的恢复办法。
pause
rem 等待用户查看程序文件缺失提示。
exit /b 1
rem 将不完整解压报告为启动失败。
:start_failed
rem 保留包内解释器输出的实际异常。
echo 启动未完成，请保留上方错误信息用于排查。
rem 使用户能看到解释器或依赖的具体失败原因。
pause
rem 防止双击启动时异常窗口立即关闭。
exit /b 1
rem 向调用方返回明确失败状态。
