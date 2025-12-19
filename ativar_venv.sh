#!/bin/bash
# Script para ativar o ambiente virtual (Git Bash / Linux / Mac)

if [ -d "venv" ]; then
    source venv/Scripts/activate
    echo "✓ Ambiente virtual ativado!"
    echo "Para desativar, digite: deactivate"
else
    echo "Erro: Ambiente virtual não encontrado!"
    echo "Execute primeiro: python -m venv venv"
fi

