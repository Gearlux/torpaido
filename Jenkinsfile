pipeline {
    agent any

    environment {
        VENV_PATH = "${WORKSPACE}/.venv"
        VENV_BIN = "${VENV_PATH}/bin"
        FORCE_JAVASCRIPT_ACTIONS_TO_NODE24 = 'true'
    }

    stages {
        stage('Initialize') {
            steps {
                echo 'Creating Isolated Virtual Environment...'
                sh "python3 -m venv ${VENV_PATH}"
                echo 'Installing Dependencies...'
                sh "${VENV_BIN}/pip install --upgrade pip"
                sh "${VENV_BIN}/pip install --no-cache-dir git+https://github.com/Gearlux/confluid.git@main"
                sh "${VENV_BIN}/pip install --no-cache-dir git+https://github.com/Gearlux/logflow.git@main"
                sh "${VENV_BIN}/pip install --no-cache-dir git+https://github.com/Gearlux/dataflux.git@main"
                sh "${VENV_BIN}/pip install -e .[dev] || ${VENV_BIN}/pip install -e ."
            }
        }

        stage('Quality Gates') {
            parallel {
                stage('Black') {
                    steps {
                        script {
                            def rc = sh(script: "${VENV_BIN}/black --check --diff torpedo tests examples > black-diff.txt 2>&1", returnStatus: true)
                            sh """${VENV_BIN}/python3 -c "
import sys, os
lines = open('black-diff.txt').readlines()
with open('black-checkstyle.xml', 'w') as f:
    f.write('<?xml version=\"1.0\" encoding=\"UTF-8\"?>\\n<checkstyle version=\"5.0\">\\n')
    for line in lines:
        if line.startswith('would reformat '):
            path = line.replace('would reformat ', '').strip()
            f.write('  <file name=\"' + path + '\">\\n')
            f.write('    <error line=\"1\" severity=\"warning\" message=\"Black would reformat this file\" source=\"black\"/>\\n')
            f.write('  </file>\\n')
    f.write('</checkstyle>\\n')
" """
                        }
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'black-torpedo',
                                name: 'Black Formatting (Torpedo)',
                                tools: [checkStyle(pattern: 'black-checkstyle.xml')]
                            )
                        }
                    }
                }
                stage('Isort') {
                    steps {
                        script {
                            def rc = sh(script: "${VENV_BIN}/isort --check-only --diff torpedo tests examples > isort-diff.txt 2>&1", returnStatus: true)
                            sh """${VENV_BIN}/python3 -c "
import sys, os
lines = open('isort-diff.txt').readlines()
with open('isort-checkstyle.xml', 'w') as f:
    f.write('<?xml version=\"1.0\" encoding=\"UTF-8\"?>\\n<checkstyle version=\"5.0\">\\n')
    for line in lines:
        if line.startswith('ERROR: '):
            path = line.split(' ')[1].strip()
            f.write('  <file name=\"' + path + '\">\\n')
            f.write('    <error line=\"1\" severity=\"warning\" message=\"Isort import order issues\" source=\"isort\"/>\\n')
            f.write('  </file>\\n')
    f.write('</checkstyle>\\n')
" """
                        }
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'isort-torpedo',
                                name: 'Isort Import Order (Torpedo)',
                                tools: [checkStyle(pattern: 'isort-checkstyle.xml')]
                            )
                        }
                    }
                }
                stage('Flake8') {
                    steps {
                        sh "${VENV_BIN}/flake8 torpedo tests examples --tee --output-file=flake8.txt || true"
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'flake8-torpedo',
                                name: 'Flake8 (Torpedo)',
                                tools: [flake8(pattern: 'flake8.txt')]
                            )
                        }
                    }
                }
                stage('Mypy') {
                    steps {
                        sh "${VENV_BIN}/mypy torpedo tests examples > mypy.txt || true"
                    }
                    post {
                        always {
                            recordIssues(
                                id: 'mypy-torpedo',
                                name: 'Mypy (Torpedo)',
                                tools: [myPy(pattern: 'mypy.txt')]
                            )
                        }
                    }
                }
            }
        }

        stage('Unit Tests') {
            steps {
                sh "${VENV_BIN}/pytest tests --junitxml=test-report.xml --cov=torpedo --cov-report=xml:coverage.xml --cov-report=term || true"
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-report.xml'
                    recordCoverage(
                        id: 'coverage-torpedo',
                        name: 'Torpedo Coverage',
                        tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']]
                    )
                }
            }
        }
    }

    post {
        always {
            echo 'Torpedo Pipeline Complete.'
        }
    }
}
