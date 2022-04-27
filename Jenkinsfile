@Library('ni-utils-private@fix-main') _

//service name is extrapolated from repository name check
def svcName = currentBuild.rawBuild.project.parent.displayName

// Define pod
def pod = libraryResource 'org/foo/infraTemplate.yaml'
def docker = libraryResource 'org/foo/dependencies/docker.yaml'
def template_vars = [
    'nodeSelectorName': 'buildnodes',
    'build_label': svcName,
    'python_version' : '3.7.13-slim',
    'image_dependencies' : [docker]
]
pod = renderTemplate(pod, template_vars)
print pod

// Define sharedLibrary
def sharedLibrary = new org.foo.pipelines.machineLearning() 

// Set slack channel
def slackChannel = "k8s-jenkins"

// Args for pipeline
def initiateData = [project: "ML"]
def compileData = [run: true, artifactType: ["DockerHub"]]
def testData = [run: false]
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
