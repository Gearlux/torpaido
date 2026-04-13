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
                
                // Internal Gearlux dependencies
                sh "${VENV_BIN}/pip install git+https://github.com/Gearlux/confluid.git@main"
                sh "${VENV_BIN}/pip install git+https://github.com/Gearlux/logflow.git@main"
                sh "${VENV_BIN}/pip install git+https://github.com/Gearlux/dataflux.git@main"
                sh "${VENV_BIN}/pip install -e .[dev]"
            }
        }

        stage('Quality Gates') {
            parallel {
                stage('Black') {
                    steps {
                        script {
                            sh "rm -f black-diff.txt black-checkstyle.xml"
                            sh "${VENV_BIN}/black --check --diff torpedo tests examples > black-diff.txt 2>&1"
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
                            sh "rm -f isort-diff.txt isort-checkstyle.xml"
                            sh "${VENV_BIN}/isort --check-only --diff torpedo tests examples > isort-diff.txt 2>&1"
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
                        sh "rm -f flake8.txt"
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
                        sh "rm -f mypy.txt"
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
                sh "${VENV_BIN}/pytest tests --junitxml=test-report.xml --cov=torpedo --cov-report=xml:coverage.xml --cov-report=term"
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-report.xml'
                    recordCoverage(
                        id: 'coverage',
                        name: 'Code Coverage',
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
        success {
            echo 'Torpedo is healthy.'
        }
        failure {
            echo 'Torpedo build failed.'
        }
    }
}
