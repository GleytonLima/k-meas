@echo off
REM Script para ativar o ambiente virtual (Windows CMD)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Ambiente virtual ativado!
    echo Para desativar, digite: deactivate
) else (
    echo Erro: Ambiente virtual nao encontrado!
    echo Execute primeiro: python -m venv venv
)

