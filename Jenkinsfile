@Library('ni-utils-test') _


//service name is extrapolated from repository name check
def svcName = (scm.getUserRemoteConfigs()[0].getUrl().tokenize('/')[3].split("\\.")[0]).toLowerCase()

// Define pod
def pod = libraryResource 'org/foo/infraTemplate.yaml'
def docker = libraryResource 'org/foo/dependencies/docker.yaml'
def alpine_curl_jq = libraryResource 'org/foo/dependencies/alpine-curl-jq.yaml'
def git = libraryResource 'org/foo/dependencies/git.yaml'
def template_vars = [
    'nodeSelectorName': 'buildnodes',
    'build_label': svcName,
    'python_version' : '3.7.13-slim',
    'image_dependencies' : [docker, alpine_curl_jq, git]
]
pod = renderTemplate(pod, template_vars)
print pod

// Define sharedLibrary
def sharedLibrary = new org.foo.pipelines.machineLearning() 

// Set slack channel
def slackChannel = "k8s-jenkins"

// Args for pipeline
def initiateData = [run: true, project: "ML", namespace: "machine-learning"]
def compileData = [run: true, artifactType: ["DockerHub"]]
def testData = [run: true]
def artifactData = [run: true, artifactType: ["DockerHub"]]
def intTestData = [run: false]
def deploymentData = [run: false, environments: ["staging"]]
def buildCommands = [
    initiateData: initiateData,
    compileData: compileData,
    testData: testData,
    artifactData: artifactData,
    intTestData: intTestData,
    deploymentData: deploymentData
]

timestamps {
    commonPipeline(sharedLibrary, svcName, buildCommands, pod, slackChannel)
}
