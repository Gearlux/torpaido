pipeline {
    agent any
    
    options {
        timeout(time: 1, unit: 'HOURS')
        timestamps()
    }
    
    stages {
        stage('Initialize') {
            steps {
                sh 'pip install -e .'
            }
        }
        stage('Lint') {
            steps {
                sh 'flake8 .'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest'
            }
        }
    }
}
